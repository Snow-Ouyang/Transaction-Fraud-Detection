# Agent handoff rules

- Read `README.md`, `docs/PROGRESS.md`, `docs/FEATURE_TIERS.md`, and `docs/VALIDATION.md` first. Current selected inputs/parameters live in `configs/current_models.json`.
- Optimize official score `50*TP - 100*FN - 5*FP`. Report auxiliary metrics without using them as an undisclosed veto. Distinguish equal-window score/10,000 from pooled total score.
- Use chronological fit/calibration/development windows. Fit category vocabularies on fit rows only; select thresholds only on preceding calibration rows. Never select features or parameters using June diagnostics or official test labels.
- All histories must be strictly earlier than the current timestamp; exclude every same-second event. No current/past fraud labels in features. Raw category and raw identity keys are excluded from the selected model inputs.
- Development windows and June have been repeatedly inspected. Do not describe them as independent, blind validation or claim leaderboard performance without an actual authorized submission.
- Current main research candidate is XGBoost15; CatBoost16 is a fixed comparison. Existing experiment scripts, protocols, locks, metrics, and manifests are historical evidence. Preserve their bytes; add new experiments in new output directories.
- For a clean clone use `scripts/run_current.py`, not the chained historical runners. Historical verifiers need locally retained models/predictions; their recorded QA is not proof of a new execution.
- New reports must be compact Markdown and machine-readable JSON/CSV. No new HTML, plots, or presentation reports unless explicitly requested. `archive/`, caches, models, and per-row predictions stay local.
- Raw CSVs are Git LFS objects, unchanged. Run `git lfs pull` and `python scripts/verify_data.py` before analysis. Do not commit credentials or install vendored dependencies into Git.
