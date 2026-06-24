# Network Sweep Report (Phase 5 Week 8)

Generated from `outputs/batch_runs/20260623T182834/batch_summary.json`. Every LeakG3PD network at or below 2000 junctions was run through the pipeline as a 24h / 1h PDD normal scenario. Validation tolerances are conservative: uncalibrated networks warn (never fail) on pressure, so only NaN/inf or a broken mass balance can fail.

**Ran:** 14 networks. **Excluded up front (WNTRSimulator-incompatible .inp):** 6.

| Network | Junc | Pipe | Tank | Resv | Demand | Runtime (s) | Severity | Worst P (m) | Mass resid (m³/s) | Notes |
|---|---:|---:|---:|---:|---|---:|---|---:|---|---|
| CalibrationNetworks.inp | 388 | 429 | 7 | 1 | fourier | 0.96 | error | n/a | n/a | excluded — NotImplementedError: Pump speeds other than 1.0 are not yet supported. |
| EPANET Net 3.inp | 92 | 117 | 3 | 2 | default | 0.53 | warning | -0.624 | 1.16e-16 | uncalibrated — sub-zero pressure (min -0.62 m), documented |
| FOWM.inp | 44 | 49 | 0 | 1 | fourier | 0.25 | ok | 0.000 | 1.11e-16 | ok — non-negative pressure, mass balance at machine epsilon |
| Hanoi.inp | 31 | 34 | 0 | 1 | fourier | 0.18 | ok | 0.000 | 3.19e-16 | ok — non-negative pressure, mass balance at machine epsilon |
| Jilin.inp | 27 | 34 | 0 | 1 | default | 0.17 | ok | 0.000 | 5.55e-17 | ok — non-negative pressure, mass balance at machine epsilon |
| LongTermImprovement.inp | 399 | 443 | 7 | 1 | default | 0.89 | error | n/a | n/a | excluded — NotImplementedError: Pump speeds other than 1.0 are not yet supported. |
| Net1.inp | 9 | 12 | 1 | 1 | fourier | 0.07 | ok | 0.000 | 1.74e-17 | ok — non-negative pressure, mass balance at machine epsilon |
| PA1.INP | 337 | 399 | 2 | 0 | default | 3.78 | ok | 6.309 | 1.53e-17 | ok — non-negative pressure, mass balance at machine epsilon |
| ky1.inp | 856 | 984 | 2 | 1 | fourier | 56.15 | error | n/a | n/a | excluded — RuntimeError: Simulation did not converge: the solver reported a non-success status (error_code=<ResultsStatus.error: 0>), commonly a max-trials-exceeded or no-solution state under PDD on an uncalibrated or ill-posed network. Treat this network as incompatible with the current simulation settings. |
| ky2.inp | 811 | 1124 | 3 | 1 | fourier | 8.53 | warning | -3.696 | 2.33e-17 | uncalibrated — sub-zero pressure (min -3.70 m), documented |
| ky3.inp | 269 | 366 | 3 | 3 | fourier | 12.38 | error | n/a | n/a | excluded — RuntimeError: Simulation did not converge: the solver reported a non-success status (error_code=<ResultsStatus.error: 0>), commonly a max-trials-exceeded or no-solution state under PDD on an uncalibrated or ill-posed network. Treat this network as incompatible with the current simulation settings. |
| ky5.inp | 420 | 496 | 3 | 4 | fourier | 16.67 | error | n/a | n/a | excluded — RuntimeError: Simulation did not converge: the solver reported a non-success status (error_code=<ResultsStatus.error: 0>), commonly a max-trials-exceeded or no-solution state under PDD on an uncalibrated or ill-posed network. Treat this network as incompatible with the current simulation settings. |
| ky8.inp | 1325 | 1614 | 5 | 2 | fourier | 45.75 | error | n/a | n/a | excluded — RuntimeError: Simulation did not converge: the solver reported a non-success status (error_code=<ResultsStatus.error: 0>), commonly a max-trials-exceeded or no-solution state under PDD on an uncalibrated or ill-posed network. Treat this network as incompatible with the current simulation settings. |
| modena.inp | 268 | 317 | 0 | 4 | fourier | 2.96 | ok | 0.000 | 7.13e-17 | ok — non-negative pressure, mass balance at machine epsilon |

## Excluded up front

These networks load correctly but are excluded from the sweep before running because the pure-Python WNTRSimulator cannot run them: either the .inp uses Darcy-Weisbach / Chezy-Manning headloss (WNTRSimulator is Hazen-Williams only) or the network is too large to simulate in reasonable time. Darcy-Weisbach-capable EpanetSimulator support (which lacks leak support) is parked in the Phase 5 backlog.

| Network | Reason |
|---|---|
| Balerma.inp | uses D-W headloss, unsupported by WNTRSimulator |
| DMAs.inp | uses D-W headloss, unsupported by WNTRSimulator |
| MarchiRural.inp | uses D-W headloss, unsupported by WNTRSimulator |
| NJ1.inp | 14991 junctions > cap 2000 (too slow for WNTRSimulator) |
| Water Sensor Network 2.inp | 12523 junctions > cap 2000 (too slow for WNTRSimulator) |
| ky17.inp | 6257 junctions > cap 2000 (too slow for WNTRSimulator) |
