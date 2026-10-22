# RUN: %PYTHON %s | FileCheck %s
"""Only eligible long axes allocate merge scratch buffers."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def sort(x):
    return torch.sort(x, stable=True)


for label, width, dtype in (
    ("MERGE", 512, torch.float32),
    ("SHORT", 511, torch.float32),
    ("F64", 512, torch.float64),
):
    compiler = DynamoCompiler(primary_registry=linalg.ops_registry)
    graph = compiler.importer(sort, torch.zeros(2, width, dtype=dtype))[0]
    graph.lower_to_top_level_ir()
    print(label)
    print(graph._imported_module)

# CHECK-LABEL: MERGE
# CHECK: memref.alloc() : memref<512xf32>
# CHECK: memref.alloc() : memref<512xi64>
# CHECK-LABEL: SHORT
# CHECK: memref.alloc() : memref<2x511xi64>
# CHECK-NOT: memref.alloc
# CHECK-LABEL: F64
# CHECK: memref.alloc() : memref<2x512xi64>
# CHECK-NOT: memref.alloc
