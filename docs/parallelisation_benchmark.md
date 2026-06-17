# Parallelisation Benchmark (Phase 5 Week 8)

Wall-clock of the batch driver running a fixed, stratified subset of 100 scale scenarios at increasing worker counts. The batch driver uses `concurrent.futures.ProcessPoolExecutor` with the `spawn` start method. One warm-up run precedes the timed runs.

## Machine

- CPU: AMD Ryzen 7 6800H with Radeon Graphics
- Logical CPUs: 4
- Platform: Linux-5.15.0-179-generic-x86_64-with-glibc2.35
- Python: 3.12.13

## Subset composition

| Type | Count |
|---|---:|
| cumulative | 12 |
| leak | 38 |
| normal | 19 |
| sensor_fault | 31 |

## Results

| Workers | Wall-clock (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 16.79 | 1.00x | 1.00 |
| 2 | 10.83 | 1.55x | 0.78 |
| 4 | 8.71 | 1.93x | 0.48 |
| 8 | 11.46 | 1.47x | 0.18 |

![speedup curve](outputs/plots/parallel_speedup.png)

## Output parity

Compared 8 scenario pressure tables between the 1-worker and 8-worker runs: max absolute difference **2.84e-14 m**, well within the 1e-12 determinism floor. Worker count does not change results.

## Commentary

Speedup tracks the ideal line up to about 4 workers (the physical core count) and then flattens: past the core count the workers oversubscribe the CPU, so additional processes contend for the same cores rather than adding throughput. The residual gap below ideal even at low worker counts comes from `spawn` process startup (a fresh Python interpreter and WNTR import per worker) and the serial tail of short scenarios. DuckDB is not exercised here (the benchmark writes parquet only); in a `--duckdb` batch the main process serialises DuckDB inserts, which adds a serial component that caps speedup further on write-heavy runs.
