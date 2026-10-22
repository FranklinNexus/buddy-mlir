# Cumulative CPU traversal

For static rank-two Cumsum (f32/i64) and Cumprod (i64) scans along axis zero
with at least 64 columns and a nonempty first axis, visit consecutive
columns together.
Cumsum starts with zero plus its first value; integer Cumprod starts with one times
its first value. Each column accumulates in
increasing row order. The previous output row supplies the accumulator;
there is no scratch allocation or reassociation. Other shapes, axes and
output types retain the existing implementation.

This targets strided loads and stores in the original column-at-a-time
loop. It adds a load of the previous output row, which is usually cache
resident. The initial operation is retained for signed-zero and NaN semantics.

## Measurement

Compare the same benchmark against baseline commit 584bf46 and the candidate,
using the same LLVM build, CPU and input seed. Run baseline, candidate,
baseline, candidate. Native timing includes output allocation and release;
export, compilation and Python tensor conversion are excluded.

```sh
python scripts/benchmark_cumsum.py --rows 1024 --width 1024 --dtype f32 --dim 0
python scripts/benchmark_cumsum.py --rows 1024 --width 1024 --dtype i64 --dim 0
```

On an i9-14900HX CPU container with four visible CPUs, PyTorch 2.10.0+cpu,
one Torch thread, and LLVM 3845a205, two rounds of median native latency
for 1024-by-1024 inputs were 4.61/4.69 ms versus 0.153/0.160 ms (f32),
and 5.43/4.38 ms versus 0.293/0.295 ms (i64). Each round used five warmups
and five batches of ten calls. These are CPU microbenchmarks, not K3 or
model-level results. Absolute timings fluctuate with the host.

Check threshold neighbors (63/64/65 columns), short and empty axes, axis
one and higher ranks as well. The unchanged path must retain identical
IR; timing fluctuations there are not evidence of an algorithm change. In this
run the unchanged f32 1024-by-1024 axis-one path measured 0.400/0.401 ms
versus 0.473/0.466 ms despite identical lowered IR. Treat these JIT timings
as subject to allocation and code-placement effects, not a speedup guarantee.

The JIT test compares against sequential additions including the initial
zero. Random floating-point sums need not match a framework implementation
that uses a different reduction order. Framework-reference tests compare
NaN positions, since NaN signs and payloads can differ, and compare all
remaining values bit for bit (including signed zeros). Experimental baseline comparisons
also cover NaNs, infinities, signed zeros, cancellation and integer overflow.

```sh
ninja -C build check-buddy
```

## Integer Cumprod

The same traversal applies to i64 Cumprod, preserving multiplication order
within each column. Shape and axis guards above still apply. Float Cumprod
keeps the original scan: folding the first multiplication by one in the
experimental path left signaling NaNs unquieted in the first output row.
That candidate was rejected despite faster finite-value benchmarks.

Use the same benchmark script on baseline 19dce0e and the candidate:

```sh
python scripts/benchmark_cumsum.py --operation cumprod --rows 1024 --width 1024 --dtype i64
```

Timing inputs are seeded random +1/-1 values to avoid overflow and
underflow. Tests separately cover integer limits and modular overflow,
floating-point special values, short/empty axes and fallback types.
The original Cumsum traversal is unchanged; IR comparisons verify this.

With the same CPU, LLVM build and timing protocol above, two rounds for
1024-by-1024 i64 Cumprod measured 3.402/3.507 ms on the baseline and
0.310/0.283 ms with contiguous traversal (about 11-12x). Float Cumprod
retains identical lowered IR and is not claimed as an optimization.
These measurements apply only to the tested CPU workloads.
