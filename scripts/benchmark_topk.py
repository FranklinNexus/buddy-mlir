"""Measure warmed CPU Top-k calls using an installed Buddy build."""

import argparse
import ctypes
import inspect
import json
import platform
import statistics
import time

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy_mlir import runtime as rt


class TopK(torch.nn.Module):
    def __init__(self, k, largest):
        super().__init__()
        self.k = k
        self.largest = largest

    def forward(self, x):
        return torch.topk(x, self.k, largest=self.largest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--smallest", action="store_true")
    parser.add_argument(
        "--pattern",
        choices=("random", "ascending", "descending", "equal"),
        default="random",
    )
    parser.add_argument(
        "--mode", choices=("native", "wrapper"), default="native"
    )
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.k <= args.width:
        parser.error("require positive rows and 1 <= k <= width")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.width)
    if args.pattern in ("ascending", "descending"):
        x = torch.arange(args.width, dtype=torch.float32).repeat(args.rows, 1)
        if args.pattern == "descending":
            x = x.flip(-1)
    elif args.pattern == "equal":
        x.zero_()
    model = TopK(args.k, not args.smallest)
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    run = compiler.dynamo_run()
    values, indices = run(x)
    torch.testing.assert_close(values, model(x).values, rtol=0, atol=0)
    torch.testing.assert_close(
        values, torch.gather(x, 1, indices), rtol=0, atol=0
    )
    assert not bool((indices.sort(1).values.diff(dim=1) == 0).any())

    if args.mode == "native":
        # Internal harness: reuse descriptors to exclude Python tensor conversion.
        closure = inspect.getclosurevars(run).nonlocals
        graph, engine = closure["graph"], closure["ee"]
        array = x.numpy().copy()
        descriptor = rt.get_ranked_memref_descriptor(array)
        input_slot = ctypes.pointer(ctypes.pointer(descriptor))
        output = graph._output_descriptor()
        output_slot = ctypes.pointer(ctypes.pointer(output))
        packed = (ctypes.c_void_p * 2)(
            ctypes.cast(output_slot, ctypes.c_void_p),
            ctypes.cast(input_slot, ctypes.c_void_p),
        )
        function = engine.lookup(graph._func_name)

        def invoke():
            function(packed)

        invoke()
        for i, expected in enumerate((values, indices)):
            actual = rt.ranked_memref_to_numpy(
                ctypes.pointer(getattr(output, str(i)))
            )
            assert actual.tobytes() == expected.numpy().tobytes()
    else:

        def invoke():
            run(x)

    for _ in range(5):
        invoke()
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(30):
            invoke()
        samples.append((time.perf_counter_ns() - start) / 30000)
    print(
        json.dumps(
            {
                **vars(args),
                "threads": 1,
                "seed": 0,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "median_us": statistics.median(samples),
                "samples_us": samples,
                "scope": "Warm-cache calls; native includes ctypes and native allocation, wrapper includes input copying and tensor conversion. JIT compilation excluded.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
