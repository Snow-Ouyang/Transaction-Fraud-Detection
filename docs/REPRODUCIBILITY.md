# Reproduction and artifact policy

## Clean clone: current selected models

1. Install Git LFS, clone the repository, then run `git lfs pull`.
2. Use Python 3.13.5 and install `requirements.txt` in a dedicated environment.
3. Run `python scripts/verify_data.py`; both source hashes must match `data/manifest.json`.
4. Run the following from the repository root. The output directory must be new for each model run.

```bash
python scripts/run_current.py --output outputs/reproduce_current --workers 2
```

The portable runner reuses the exact historical feature builders, score/threshold functions, and model adapters. It needs no previous model or feature-cache files. Current inputs and hyperparameters are frozen in `configs/current_models.json`; it does not repeat hyperparameter search or feature selection.

Handoff validation on 2026-10-21 completed all six fresh fits in the recorded environment. The fully rebuilt feature cache matched the historical cache exactly; all calibration/development prediction rows matched exactly, and full June scores matched within 1e-12. In total 1,440,010 prediction rows were compared. Both model families reproduced every recorded development score and June score. A separate clean-environment pip installation was not performed. Machine-readable evidence: [reproduction_results.json](reproduction_results.json).

Features are reconstructed from `fraudTrain.csv` only, including literal historical spot audits and cutoff-prefix checks. Two concurrent processes each use the existing four-thread model settings. Complete calibration/development predictions, models and schemas remain in the ignored output directory. June is evaluated only after all three development folds finish. It compares the resulting development and June scores to the recorded reference values without changing thresholds or features.

Useful variants:

```bash
python scripts/run_current.py --output outputs/features_check --prepare-only
python scripts/run_current.py --output outputs/xgb_check --family xgboost --workers 2
python eda/run_eda.py --mode analyze
```

`eda/run_eda.py --mode analyze` recomputes machine-readable statistics from both source CSVs; it does not fit a model. It rewrites tracked EDA result files, so use a fresh working branch/copy if comparing output. The historical rendering mode is not part of the supported Markdown workflow.

## Historical research evidence

`models/<experiment>/` contains original scientific Python, fixed protocols, decision locks, manifests, QA records, tables and compact per-run `result.json`/`June_result.json`. Source bytes are preserved, including legacy optional report code, so recorded source hashes retain meaning. README files are refreshed for the GitHub handoff.

Historical runners often depend on prior models, per-row predictions, exact original feature caches and old report paths. These are retained on the owner's machine but excluded from Git. Their historical QA records describe completed original runs; they cannot be rerun wholesale on a lean clone without regenerating the dependent artifacts. Use the new portable entry point for the current models.

## What is and is not versioned

- Git LFS: the two unmodified raw CSVs, at their original root paths.
- Regular Git: Python source, Markdown documentation, requirements, configuration, data hashes, machine-readable statistical tables, protocols, selected-decision locks, compact metrics and QA evidence.
- Local only: old HTML/illustrated Markdown reports, figures, presentation QA images, original project PDF, trained models, feature caches, per-row predictions, `.deps`, scratch and new `outputs/`.
- `archive/pre_github_2026-10-21/` is a non-destructive snapshot of historical presentation files and original README files. Ignored originals remain at their historical paths for local frozen checks; neither copy is uploaded.

No official test predictions or competition submissions are part of this handoff. Future experiments should create new output directories and compact Markdown/JSON records; preserve the fixed historical evidence.
