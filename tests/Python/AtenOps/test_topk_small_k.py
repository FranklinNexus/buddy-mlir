# RUN: %PYTHON %s | FileCheck %s
"""The small-k path avoids an input-sized visited bitmap; fallback retains it."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def small(x):
    return torch.topk(x, 8)


def fallback(x):
    return torch.topk(x, 9)


for label, function in (("SMALL", small), ("FALLBACK", fallback)):
    compiler = DynamoCompiler(primary_registry=linalg.ops_registry)
    graph = compiler.importer(function, torch.randn(3, 17))[0]
    graph.lower_to_top_level_ir()
    print(label)
    print(graph._imported_module)

# CHECK-LABEL: SMALL
# CHECK: memref.alloc() : memref<3x8xf32>
# CHECK: memref.alloc() : memref<3x8xi64>
# CHECK-NOT: memref.alloc
# CHECK: scf.if
# CHECK-NOT: memref.alloc
# CHECK-LABEL: FALLBACK
# CHECK: memref.alloc() : memref<3x9xf32>
# CHECK: memref.alloc() : memref<3x9xi64>
# CHECK: memref.alloc() : memref<3x17xi1>
