# Small-k CPU Top-k

The Linalg lowering uses a bounded insertion scan for static rank-2 float32
inputs selected along the last axis with `2 <= k <= 8`. It initializes k sorted
slots, scans each remaining input once, and inserts only values that improve the
last slot. A value strictly better than the first slot shifts the slots directly.
Worst-case work is still proportional to axis length times k; the benefit is
avoiding repeated input scans and the input-sized visited bitmap.
Other ranks, axes, dtypes and k values retain repeated selection. Empty outputs
and scalar inputs keep their existing handling.

The optimization changes selection and data movement, not arithmetic. NaNs rank
above numbers; equal values retain their earlier input index, including displaced
ties. Both `largest` modes are supported. Results remain sorted when
`sorted=False`, as permitted by PyTorch. The output values and indices are checked
bit for bit against the previous lowering in the validation experiment.

## Reproduce

Build Buddy with Python bindings, then set `PYTHONPATH` and `LLVM_LIBS_DIR` for
that build as described in [PyTorch operator coverage](PytorchOpCoverage.md).
Run the internal benchmark from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark_topk.py \
  --rows 128 --width 256 --k 8 --mode native
```

Use `--mode wrapper` to include Python input copying and output conversion.
Also test `--pattern ascending`, `descending`, `equal`, and `--smallest`.
Each process uses seed 0, one Torch thread, five warmups and seven samples of
30 invocations. Compilation is excluded. Native mode reuses input and output
descriptors but still includes native output allocation and the ctypes call.
The JIT wrapper retains native output allocations, so run these bounded samples
in separate processes; this is not a long-running memory benchmark.

Build baseline and candidate against the same LLVM revision and dependencies.
Run baseline, candidate, baseline, candidate on the same idle machine. Record
both Git revisions, CPU, thread count and all samples. Do not time concurrently
with builds or tests. Small timing differences within the observed spread are
not evidence of a speedup. These CPU microbenchmarks do not establish model-level
or RISC-V hardware performance.

## CPU measurement

On an Intel Core i9-14900HX, Linux x86-64 (four visible CPUs), Python 3.12.3
and PyTorch 2.10.0+cpu, the baseline lowering at `953f9bf8` and this lowering
were compiled with the same LLVM revision `3845a205` and JIT pipeline.
Fifteen shape/order/direction cases produced bit-identical values and indices.
For random float32 `[128, 256]`, k=8, the two alternating runs measured:

- Native calls: baseline 401.67 / 395.11 us; candidate 81.94 / 90.40 us.
- Python wrapper: baseline 560.53 / 533.00 us; candidate 248.33 / 242.53 us.

These are medians of seven batches per run, not a model-level speedup.
Across both runs, native batch times ranged from 380.65 to 427.11 us for
the baseline and 79.71 to 117.70 us for the candidate.
Ascending rows took 360.76–365.60 us before and 120.72–121.18 us after;
descending and equal rows also improved. Narrow `[128, 17]`, k=8 rows had
more variable candidate medians (13.90–16.77 us). The unchanged k=1 and k=9
fallbacks showed timing variation, not a demonstrated improvement.
The lowered LLVM IR reduced allocation calls from three to two by removing
the visited bitmap. Re-measure on the intended hardware before expanding
the specialization.

## Validation

```bash
ninja -C build check-buddy
```

`tests/Python/AtenOps/test_topk_small_k.py` checks that the small-k lowering avoids
the input-sized visited bitmap and that k=9 retains it. The JIT small-k tests
cover all k values from 2 through 8, both selection directions, equal values,
NaNs, infinities, signed zero and odd axis lengths. Existing Top-k tests retain
coverage of integer/f64 inputs, other axes, internal strided views, empty outputs
and scalar inputs.
