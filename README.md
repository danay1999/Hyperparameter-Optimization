# Hyperparameter Optimization

Compare four search methods across 51 dataset–condition–model cells. Shared Python code runs the experiments; notebooks explore data and analyze saved results. See [the protocol](docs/experiments.md) and [the project proposal](Project%20Proposal.md).

## Setup and pilot

Run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m ipykernel install --user --name=.venv --display-name "MAP6114 Python (.venv)"
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --pilot
jupyter notebook notebooks/02_pilot_results.ipynb
```

The pilot uses synthetic linear regression, Elastic Net, all four optimizers, seed 1, three CV folds, and budget 4. It requires no dataset download. The full manifest has 51 cells and 3,060 runs. Select subsets with `--dataset`, `--condition`, `--model`, `--optimizer`, and `--seed`.

## Structure

- `configs/`: experimental matrix and provisional shared search ranges.
- `src/hpo_project/`: dataset loading, fold-fitted preprocessing, model factories, optimizer adapters, runner, analysis.
- `scripts/run_experiments.py`: batch execution and dry-run inventory.
- `notebooks/`: data exploration, pilot results, final analysis starters.
- `results/runs/`: one JSON per run with configuration, provenance, trial history, metrics, and status; excluded from Git.
- `figures/`: exported plots; excluded from Git.

Completed compatible experiment combinations are skipped on rerun, regardless of filename, code hash, or Git revision. Failed and interrupted runs are saved and restarted from the beginning; per-trial checkpoints preserve diagnostics but do not resume individual trials. Result files are written atomically after each trial and at completion. IDs include configuration, Git revision, and source hash. RS/BO derive both budgets from one trial history, refitting the best configuration for each budget separately. Scores in trial histories are maximization scores (negative RMSE for regression); summaries use positive RMSE. Generalization gaps are oriented so positive means worse test performance.

## Before the full study

This is a runnable foundation, not a frozen benchmark:

- Review and freeze search ranges and dataset-specific cleaning against downloaded schemas.
- Pilot LightGBM currently uses the shared one-hot preprocessing. Native categorical preprocessing remains to implement before matching the documented protocol.
- Halving uses three rungs with one-third elimination, fixed validation folds, and nested training subsets; LightGBM uses boosting rounds. Resource totals are logged as full-fidelity equivalents. This is a custom sequential adapter, not `HalvingRandomSearchCV`; calibrate schedules for budgets 25 and 100 and compare measured times before freezing the protocol. Boosting rounds are fixed by the resource schedule rather than independently searched.
- Grid search grows a rectangular grid up to the budget; high-dimensional spaces necessarily leave some parameters at one value. Review those values and parameter priorities before benchmarking.
- Classification PR performance is currently reported as average precision, explicitly named in the saved metrics.
- Run the one-seed timing survey before launching all seeds. Statistical comparisons and publication plots remain to implement in the final analysis notebook.

Use one batch process per results directory. Run-level resume is supported; simultaneous writers and trial-level resume are not supported.

## macOS LightGBM dependency

LightGBM requires the OpenMP runtime. For an Apple Silicon Python environment, install it with the Apple Silicon Homebrew executable:

```sh
/opt/homebrew/bin/brew install libomp
.venv/bin/python -c "import lightgbm; print(lightgbm.__version__)"
```

An Intel OpenMP library under `/usr/local` cannot satisfy an Apple Silicon LightGBM installation. On an Intel Mac, use `brew install libomp` instead. After fixing the dependency, rerun the same experiment command; completed runs are skipped and the failed run is retried.

## Convergence diagnostics

Regression Elastic Net uses `max_iter=50000` and a shared `l1_ratio` range of 0.05–1.0. Classification settings are unchanged. Each trial records `converged` and `convergence_warnings`, including parallel CV workers; summaries record `best_trial_converged`, `refit_converged`, and `refit_convergence_warnings`. Trials with warnings remain in the search and should be reviewed before reporting results. Absence of convergence warnings is the criterion for these flags.

Restart a running batch to load code or configuration changes. Source changes produce new run IDs as provenance, but compatible completed combinations are reused. Keep analysis restricted to one `source_hash` and a consistent protocol; the current analysis loader includes all completed files, including older runs. Older files have no convergence diagnostics and should be treated as unknown, not converged.

## Resume and CPU settings

The default is `cv_jobs: 4` and `model_threads: 1`: up to four fold workers, each with one native thread. All four optimizers, including halving, use parallel CV. Trials remain sequential. Change settings in the configuration or pass `--cv-jobs` and `--model-threads`.

```sh
# Review remaining seed-1 work, explicitly accepting older model metadata:
python scripts/run_experiments.py --seed 1 --dry-run --accept-legacy-results
# Resume seed 1:
python scripts/run_experiments.py --seed 1 --accept-legacy-results
# Deliberately repeat a selected combination, preserving older files:
python scripts/run_experiments.py --dataset synthetic_linear --model elastic_net --optimizer random --seed 1 --force
```

Resume requires the same dataset settings, search space, CV folds, protocol version, package versions, and (for new files) model settings. It also checks that all requested budget summaries and trial history exist. Parallel execution settings are recorded but do not invalidate predictive scores; timing comparisons must separate worker/thread settings. Expanding the requested budgets may rerun a random/Bayesian search if its older file lacks a requested summary.

`--accept-legacy-results` is an explicit decision to trust missing model-setting metadata in older results. It bypasses only that missing metadata check, not other compatibility checks. Without this option, those combinations are rerun. Historical files may include different Elastic Net iteration limits or equivalent logistic APIs; audit them before making final statistical claims.

`--dry-run` shows skip reasons and remaining counts without fitting or modifying results. `--force` bypasses skips and keeps previous files; duplicate outputs must be selected deliberately in analysis. Future scientific changes to preprocessing, split generation, or optimizer semantics must increment `protocol_version` in the runner; a source hash alone no longer controls reuse.
