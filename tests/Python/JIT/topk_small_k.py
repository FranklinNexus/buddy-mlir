# RUN: %PYTHON %s
"""Small-k insertion preserves stable indices, NaNs and signed zero."""

import math

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class TopK(torch.nn.Module):
    def __init__(self, k, largest):
        super().__init__()
        self.k = k
        self.largest = largest

    def forward(self, x):
        return torch.topk(x, self.k, dim=-1, largest=self.largest)


torch.set_num_threads(1)
generator = torch.Generator().manual_seed(42)
count = 0
for width in (8, 17, 64):
    for k in range(2, 9):
        for largest in (False, True):
            model = TopK(k, largest)
            example = torch.zeros(3, width)
            exported = torch.export.export(model, (example,), strict=True)
            compiler = DynamoCompiler(
                primary_registry=tosa.ops_registry, enable_external_calls=False
            )
            compiler._compile_fx(exported.graph_module, [example])
            run = compiler.dynamo_run()
            special = torch.tensor(
                [float("nan"), -0.0, 0.0, -float("inf"), float("inf"), 2, 2]
            )
            cases = [
                example,
                torch.full_like(example, float("nan")),
                torch.arange(width).float().repeat(3, 1),
                -torch.arange(width).float().repeat(3, 1),
                special.repeat((3 * width + 6) // 7)[: 3 * width].reshape(
                    3, width
                ),
            ]
            cases += [
                torch.randint(-4, 5, (3, width), generator=generator).float()
                for _ in range(10)
            ]
            for x in cases:
                original = x.clone()
                values, indices = run(x)
                assert values.dtype == torch.float32
                assert indices.dtype == torch.int64
                assert values.shape == indices.shape == (3, k)
                torch.testing.assert_close(
                    values, model(x).values, rtol=0, atol=0, equal_nan=True
                )
                for row in range(3):

                    def key(i, x=x, row=row, largest=largest):
                        value = float(x[row, i])
                        if math.isnan(value):
                            return (0 if largest else 1, 0)
                        return (
                            1 if largest else 0,
                            -value if largest else value,
                        )

                    # Python's stable sort gives an independent first-index oracle.
                    expected = sorted(range(width), key=key)[:k]
                    assert indices[row].tolist() == expected
                    assert (
                        values[row].numpy().tobytes()
                        == x[row, expected].numpy().tobytes()
                    )
                assert x.numpy().tobytes() == original.numpy().tobytes()
                count += 1
print(f"Small-k: {count} cases passed")
