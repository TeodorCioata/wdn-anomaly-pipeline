"""Downstream ML-utility experiments over the generated WDN datasets.

This package is *evaluation code that consumes the dataset*, not part of the
simulation pipeline. It reads data exclusively through the public
:class:`wdn_pipeline.query.DatasetQuery` and the PyTorch loader in
:mod:`wdn_pipeline.ml`, and it never modifies ``src/wdn_pipeline`` behaviour.

The scripts produce the evidence for Section 5 of the journal paper:

- :mod:`experiments.ml_utility.generate_experiment_dataset` builds a dedicated,
  difficulty-swept dataset (Study 4 data).
- :mod:`experiments.ml_utility.detectors` defines the three anomaly detectors.
- :mod:`experiments.ml_utility.metrics` defines the evaluation metrics.
- :mod:`experiments.ml_utility.run_experiments` trains, evaluates and plots
  (Study 4 results).
- :mod:`experiments.ml_utility.study5_leakg3pd` is the LeakG3PD cross-check.
- :mod:`experiments.ml_utility.refresh_supporting_studies` refreshes Studies 1-3.

Every script fixes and records its seeds so the paper numbers are reproducible.
"""
