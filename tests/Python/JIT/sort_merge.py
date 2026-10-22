# RUN: %PYTHON %s
"""Stable merge ordering, odd tails, fallback types and strided source views."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Sort(torch.nn.Module):
    def __init__(self, dim, descending, indices_only=False, transpose=False):
        super().__init__()
        self.dim = dim
        self.descending = descending
        self.indices_only = indices_only
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.T
        if self.indices_only:
            return torch.argsort(
                x, dim=self.dim, descending=self.descending, stable=True
            )
        return torch.sort(
            x, dim=self.dim, descending=self.descending, stable=True
        )


def check(model, example):
    exported = torch.export.export(model, (example,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [example])
    run = compiler.dynamo_run()
    cases = [example, torch.zeros_like(example)]
    if example.dtype.is_floating_point:
        special = torch.tensor(
            [float("nan"), -0.0, 0.0, float("inf"), -float("inf"), 2, 2],
            dtype=example.dtype,
        )
        cases.append(
            special.repeat((example.numel() + 6) // 7)[
                : example.numel()
            ].reshape(example.shape)
        )
    for x in cases:
        original = x.numpy().tobytes()
        expected = model(x)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        actual = run(x)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert (
                value.shape == reference.shape
                and value.dtype == reference.dtype
            )
            assert value.numpy().tobytes() == reference.numpy().tobytes()
        assert x.numpy().tobytes() == original
    return len(cases)


torch.set_num_threads(1)
torch.manual_seed(42)
count = 0
for dtype in (torch.int64, torch.float32, torch.int32, torch.float64):
    for shape, dim in (
        ((511,), 0),
        ((512,), 0),
        ((513,), 0),
        ((2, 513), 1),
        ((513, 2), 0),
        ((2, 513, 2), 1),
        ((0, 513), 1),
    ):
        for descending in (False, True):
            x = torch.randint(-4, 5, shape).to(dtype)
            count += check(Sort(dim, descending), x)
for dtype in (torch.int64, torch.float32):
    for descending in (False, True):
        for indices_only in (False, True):
            x = torch.randint(-4, 5, (2, 513)).to(dtype)
            count += check(Sort(0, descending, indices_only, transpose=True), x)
print(f"Stable merge: {count} cases passed")
