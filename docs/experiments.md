# Experiments

This document defines every combination of dataset, data condition, model, optimizer, and budget the project will run. Datasets are described in [datasets.md](datasets.md) and models in [models.md](models.md).

## Experimental factors

| Factor | Levels |
| --- | --- |
| Dataset | 6 real (D1–D6) + 4 synthetic (S1–S4) |
| Data condition | Base, Small (subsampled), Noisy |
| Model | Elastic Net (EN), SVM with RBF kernel (SVM), LightGBM (LGBM) |
| Optimizer | Grid search (GS), Random search (RS), Bayesian optimization (BO), Successive halving (SH) |
| Budget | 25 and 100 full-fidelity evaluations |
| Repetitions | 10 seeds |

## Datasets and conditions

### Real datasets

| ID | Dataset | Task | Base rows | Conditions run |
| --- | --- | --- | --- | --- |
| D1 | White Wine Quality | Regression | 4,898 | Base, Noisy |
| D2 | Superconductivity | Regression | 21,263 | Base, Small, Noisy |
| D3 | Communities and Crime | Regression | 1,994 | Base, Noisy |
| D4 | Bank Marketing | Classification | 45,211 | Base, Small, Noisy |
| D5 | Bioresponse | Classification | 3,751 | Base, Noisy |
| D6 | Default of Credit Card Clients | Classification | 30,000 | Base, Small, Noisy |

Only the large datasets (D2, D4, D6) get a **Small** condition, because the others are already small.

- **Small:** stratified random subsample of 2,000 rows (stratified on the class for classification, on target quantiles for regression). 2,000 roughly matches the size of D3.
- **Noisy (regression, D1–D3):** Gaussian noise with standard deviation 0.5 × SD(y) added to the target.
- **Noisy (classification, D4–D6):** 10% of labels flipped at random.
- Noise is added to the **training data only**. The test set stays clean, so test scores measure how well the tuned model recovers the true signal.

### Synthetic datasets

| ID | Generator | Task | Settings |
| --- | --- | --- | --- |
| S1 | `make_regression` | Regression, linear | n = 2,000, 20 features (10 informative), noise = 10 |
| S2 | `make_friedman1` | Regression, nonlinear | n = 2,000, 20 features (5 informative), noise = 1 |
| S3 | `make_classification` | Classification, balanced | n = 2,000, 20 features (5 informative, 5 redundant), `flip_y` = 0.01 |
| S4 | `make_classification` | Classification, imbalanced | Same as S3 with `weights` = [0.9, 0.1] |

Synthetic datasets are run in the Base condition only. Their settings can be varied later to isolate single factors.

## Models per dataset

- **EN** and **LGBM** run on every dataset and condition.
- **SVM** runs only when the dataset has at most 5,000 rows. Its training cost grows at least with the square of the number of rows, so the large Base and Noisy versions of D2, D4, and D6 are excluded.

## Optimizers

All four optimizers use the same search space, data split, cross-validation folds, and budget within each run.

| ID | Optimizer | Suggested implementation | How it uses the budget |
| --- | --- | --- | --- |
| GS | Grid search | `sklearn.model_selection.GridSearchCV` | A grid with at most *B* points. When the grid cannot cover every hyperparameter, it covers the most important ones and fixes the rest at defaults. |
| RS | Random search | `sklearn.model_selection.RandomizedSearchCV` | *B* configurations sampled from the search space. |
| BO | Bayesian optimization | Optuna `TPESampler` | *B* sequential trials. |
| SH | Successive halving | `sklearn.model_selection.HalvingRandomSearchCV` | Candidates and elimination factor chosen so the total resource spent is about *B* full-fidelity evaluations. The resource is training sample size for EN and SVM, and boosting rounds for LGBM. |

## Budgets

The budget *B* is the number of **full-fidelity evaluations**, where one evaluation is one configuration scored by full 5-fold cross-validation. Two levels are used:

- **B = 25**: low budget
- **B = 100**: high budget

Wall-clock time is recorded for every trial so results can also be compared at equal elapsed time.

For RS and BO, the B = 25 result is the first 25 trials of the B = 100 run. Neither method depends on the total budget, so they are run once at B = 100. GS and SH change their design with the budget, so they are run separately at each level.

## Protocol for each run

1. **Split:** 80% train / 20% test, stratified for classification. The split is fixed by the repetition seed and shared by all optimizers and models in that repetition.
2. **Tuning:** 5-fold cross-validation (stratified for classification) on the training set, with the same folds for all optimizers.
3. **Tuning objective:** RMSE for regression, ROC-AUC for classification.
4. **Refit:** the best configuration is refit on the full training set and scored once on the test set.
5. **Seeds:** repetition seeds 1–10 set the train/test split, CV folds, subsampling, noise, and optimizer randomness.
6. **Hardware:** all runs use the same machine type and the same `n_jobs` setting so that times are comparable.

### Logged for every run

- Best validation score and test score (generalization gap = validation − test)
- Secondary metrics on the test set: MAE (regression); F1, PR-AUC, and log loss (classification)
- For every trial: configuration, CV score, resource used, and elapsed time, to build best-so-far curves against both number of trials and time
- Total optimization time and number of configurations evaluated

## Run matrix

Each ✓ is one dataset–condition–model cell. Every cell runs all four optimizers at both budgets for 10 seeds.

| Dataset | Condition | Rows | EN | SVM | LGBM | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| D1 White Wine | Base | 4,898 | ✓ | ✓ | ✓ | 3 |
| D1 White Wine | Noisy | 4,898 | ✓ | ✓ | ✓ | 3 |
| D2 Superconductivity | Base | 21,263 | ✓ | — | ✓ | 2 |
| D2 Superconductivity | Small | 2,000 | ✓ | ✓ | ✓ | 3 |
| D2 Superconductivity | Noisy | 21,263 | ✓ | — | ✓ | 2 |
| D3 Communities and Crime | Base | 1,994 | ✓ | ✓ | ✓ | 3 |
| D3 Communities and Crime | Noisy | 1,994 | ✓ | ✓ | ✓ | 3 |
| D4 Bank Marketing | Base | 45,211 | ✓ | — | ✓ | 2 |
| D4 Bank Marketing | Small | 2,000 | ✓ | ✓ | ✓ | 3 |
| D4 Bank Marketing | Noisy | 45,211 | ✓ | — | ✓ | 2 |
| D5 Bioresponse | Base | 3,751 | ✓ | ✓ | ✓ | 3 |
| D5 Bioresponse | Noisy | 3,751 | ✓ | ✓ | ✓ | 3 |
| D6 Credit Default | Base | 30,000 | ✓ | — | ✓ | 2 |
| D6 Credit Default | Small | 2,000 | ✓ | ✓ | ✓ | 3 |
| D6 Credit Default | Noisy | 30,000 | ✓ | — | ✓ | 2 |
| S1 Linear regression | Base | 2,000 | ✓ | ✓ | ✓ | 3 |
| S2 Friedman #1 | Base | 2,000 | ✓ | ✓ | ✓ | 3 |
| S3 Balanced classification | Base | 2,000 | ✓ | ✓ | ✓ | 3 |
| S4 Imbalanced classification | Base | 2,000 | ✓ | ✓ | ✓ | 3 |
| **Total** | | | **19** | **13** | **19** | **51** |

## Run count

| Quantity | Calculation | Total |
| --- | --- | --- |
| Dataset–condition–model cells | from the run matrix | 51 |
| Optimization runs per cell per seed | GS ×2, SH ×2, RS ×1, BO ×1 | 6 |
| Optimization runs | 51 × 6 × 10 seeds | 3,060 |
| Full-fidelity evaluations per cell per seed | GS (25 + 100) + SH (25 + 100) + RS 100 + BO 100 | 450 |
| Model fits (5-fold CV) | 51 × 450 × 10 × 5 | about 1.15 million |

Most fits are on 2,000–5,000 rows and take under a second. Most of the compute goes to the 12 large cells (D2, D4, D6 in the Base and Noisy conditions, EN and LGBM).

**Before the full run,** we run one seed of every cell to measure time. If the total is too large, we cut compute in this order:

1. Reduce seeds from 10 to 5 for the large cells only.
2. Use 3-fold instead of 5-fold cross-validation for the large cells.
3. Drop the Noisy condition for D4 and D6 (the noise effect is still measured on D1–D3 for regression and on D5 for classification).

## Optional extension: search-space dimension

To separate the effect of search-space size from the effect of model family, LGBM can also be tuned with nested search spaces on a subset of datasets (for example D2-Small, D5, and D6-Small):

| Search space | Tuned hyperparameters |
| --- | --- |
| LGBM-2 | `learning_rate`, `num_leaves` |
| LGBM-4 | LGBM-2 + `min_child_samples`, `reg_lambda` |
| LGBM-10 | full search space from [models.md](models.md) |

This adds 3 datasets × 2 extra search spaces = 6 cells (LGBM-10 is already in the main matrix).
