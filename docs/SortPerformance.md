# Stable long-axis Sort and Argsort

The Linalg lowering uses bottom-up stable merge sort for static float32 and
int64 axes of length at least 512. Short axes and other types retain adjacent
swaps. Both directions, arbitrary axes and Argsort use the same ordering:
NaNs rank above numbers, and equal values (including signed zero) keep their
input order. No floating-point arithmetic is introduced.

Each merge pass writes to axis-sized value/index scratch buffers and copies
back to the output. The pair is reused across serial rows, using O(axis length)
extra storage and O(axis length * log(axis length)) work per row. The existing
ownership-based deallocation pipeline releases both scratch buffers. Exhausted
merge cursors are clamped before eager loads, including odd trailing runs.
At axis length 1024, adjacent swaps execute 523,776 pair comparisons per row;
ten merge passes visit 10,240 elements. This explains the optimization target,
but scratch copies and allocation prevent treating that ratio as a speedup.

## Reproduce

Build Buddy's Python bindings and set the environment described in
[PyTorch operator coverage](PytorchOpCoverage.md). From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark_sort.py \
  --rows 8 --width 1024 --dtype i64
```

Use `--indices-only` for Argsort, `--mode wrapper` to include Python conversion,
`--descending` for descending output, and `--pattern ascending`, `descending`
or `equal` for input distributions. Random inputs use seed 0 and repeated
integers in [-32, 32], cast to the requested dtype. The script validates stable
values/indices before timing. Each process takes five batches of ten calls
after five warmups; compilation is excluded. Native calls still include ctypes
and allocation. Output allocations are retained by the JIT wrapper, so use
bounded runs rather than a long-running memory test.

Compare baseline/candidate/baseline/candidate on the same idle machine, LLVM
revision and dependencies. Record both source revisions and the JSON samples.
Run this same script against both installed builds.
Test threshold neighbors (511, 512, 513) and odd large lengths as well as powers
of two. The threshold is conservative: experiments at lengths 64 and 65 slowed
already sorted or equal inputs despite speeding up random inputs.

## CPU results

On an Intel Core i9-14900HX under Linux x86-64, with Python 3.12.3,
PyTorch 2.10.0+cpu, one Torch thread and LLVM `3845a205`, the baseline at
`088cf89` and this lowering were tested with the same JIT and native ABI.
Two alternating runs of five batches of ten calls gave these medians for
random [8, 1024] inputs (microseconds):

- int64 Sort: 6638.47 / 6684.54 before; 236.23 / 240.61 after.
- float32 Sort: 7041.60 / 7302.62 before; 430.41 / 435.11 after.
- int64 Argsort: 6466.82 / 6589.52 before; 207.65 / 211.17 after.
- float32 Argsort: 6954.73 / 7011.79 before; 392.32 / 394.79 after.

The 48 Sort comparisons cover both types, lengths 256/511/512/513/1024/1025,
and random/sorted/reversed/equal inputs. All values and indices matched the
baseline and PyTorch bit for bit. Long-axis distributions improved; fallback
timing differences are noise, not claimed speedups. LLVM IR has four allocation
calls and two frees versus two allocations in the baseline. These are CPU
microbenchmarks, not model-level or K3 performance results.

Twelve additional comparisons cover random and sorted [1025], [512, 8] along
axis 0, and [2, 513, 4] along axis 1, for both types. Results were bit-identical
and both candidate median timings were lower in each case; the benefit for
sorted non-last integer axes was smaller than for random last-axis inputs.

## Validation

Run `ninja -C build check-buddy`. `tests/Python/JIT/sort_merge.py` checks threshold
edges, odd tails, both directions, repeated values, NaNs, infinities, signed zero,
empty batches, non-last axes, internal transposed views, Argsort and fallback
types. `tests/Python/AtenOps/test_sort_merge.py` checks scratch allocation only
for eligible axes. Existing sort tests cover scalar and other short-axis cases.
