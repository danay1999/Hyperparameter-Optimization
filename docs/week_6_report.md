# Week 6 Project Report

*Implementation update: October 30, 2026. The Week 6 meeting information below is retained; project details have been revised to describe what was built.*

## 1. Team Information

- **Team name:** Optima Prime
- **Team members:**
  - Danay Fernandes Alfonso
  - Dilek Ozdemirci
  - Ecil Teodoro

## 2. Team Coordination

- **Week 6 meeting date:** Sat, Sep 26, 2026
- **Meeting time:** 1pm EDT
- **Communication methods:** Zoom

During the meeting, we discussed the project scope, possible datasets, how we will divide responsibilities, and our implementation approach. We will continue to use Zoom to meet, coordinate development, share results, and track project progress.

## 3. Project Details

### Project Idea

Our main question is whether the best way to tune a model changes when the model, the data, or the available computing budget changes. We implemented grid search, random search, Bayesian optimization, and successive halving on three models with small, medium, and large search spaces. The benchmark covers six public datasets and four synthetic tasks. The analysis is designed to identify situations in which a simple search is sufficient and situations in which a more adaptive method is worth the extra complexity.

### Dataset

We use six public datasets from OpenML together with synthetic datasets generated with Scikit-learn. We picked them so that, as a group, they include regression and classification problems; small and large sample sizes; few and many features; balanced and imbalanced classes; clean and noisy data; and numeric as well as categorical features.

| Dataset | OpenML ID | Task | Rows x Features | Notes |
| --- | --- | --- | --- | --- |
| White Wine Quality | 44971 | Regression | 4,898 x 11 | Small, few features |
| Superconductivity | 44964 | Regression | 21,263 x 81 | Large, nonlinear |
| Communities and Crime | 46286 | Regression | 1,994 x 122 | Small, many features, noisy |
| Bank Marketing | 1461 | Classification | 45,211 x 16 | Large, imbalanced, categorical features |
| Bioresponse | 4134 | Classification | 3,751 x 1,776 | Small, very many features, balanced |
| Default of Credit Card Clients | 42477 | Classification | 30,000 x 23 | Large, imbalanced, noisy labels |
| Synthetic | N/A | Both | 2,000 x 20 | Linear and nonlinear, balanced and imbalanced |

The synthetic data gives us a baseline where we know the true relationship between the predictors and the target. We implemented modified training conditions for the real datasets. After holding out a clean 20% test set, the Small condition subsamples the training set to 2,000 observations for the three large datasets. The Noisy condition adds Gaussian regression-target noise with standard deviation 0.5 times the training target standard deviation, or flips 10% of classification labels. The test set stays unchanged across conditions for a dataset and seed.

Two datasets changed from our proposal. Madelon turned out to be synthetic (Scikit-learn’s make_classification function uses the same generator), so it wouldn't add anything our synthetic data doesn't already cover. We replaced it with Bioresponse, which is real data with a very large number of features. We also dropped Census-Income KDD because, at almost 300,000 rows, tuning it many times would take too long. Default of Credit Card Clients takes its place. More details are in [datasets.md](datasets.md).

### Models

We changed the models as well. Ridge and logistic regression only have one or two hyperparameters, so there is not much for the different search methods to disagree on. We still wanted a simple linear model, so we moved to Elastic Net, and we added two models with more hyperparameters:

| Search space | Model | Hyperparameters tuned |
| --- | --- | --- |
| Small (2-3) | Elastic Net (regression) and logistic regression with elastic-net penalty (classification) | Regularization strength, L1/L2 mix, class weights |
| Medium (3) | Support vector machine with RBF kernel | C, gamma, and epsilon or class weights |
| Large (10) | LightGBM | Learning rate, number of trees, tree size, sampling rates, regularization |

Comparisons across these model families show whether optimizer performance depends on the model, but do not isolate the effect of search-space dimension because model properties also change. LightGBM replaces Scikit-learn’s gradient-boosted trees since it trains faster and has more settings to tune. More details are in [models.md](models.md).

### Methodology

We use shared search spaces, an 80/20 train/test split, and five-fold cross-validation on the training portion, stratified for classification. The implemented budgets are 25 and 100 full-fidelity CV evaluations, with elapsed time measured separately. Resource matching is approximate for halving and does not imply equal runtime. **We reduced the original ten-seed plan to five seeds (1–5) for every cell because of compute time.** We retained five CV folds and both budgets.

Random and Bayesian search produce the budget-25 result from the first 25 trials of the budget-100 history, refitting separately for each budget. Grid and halving run separately at each budget. The five-seed design targets 1,530 optimization runs and 2,040 optimizer–budget–seed summaries across 51 cells. These are design totals; historical runs and incomplete compatible coverage mean saved-file counts differ.

The runner uses custom adapters built around Scikit-learn estimators, parameter grids and sampling, and Optuna TPE. Successive halving has three resource rungs (1/9, 1/3, and 1), retains approximately one third of candidates per rung, and uses fixed validation folds with nested training subsets. LightGBM uses boosting rounds as its resource, so its tree count is scheduled rather than independently searched. Grid search evaluates a rectangular grid with at most the requested budget; some parameters receive only one value in large spaces.

We tune for RMSE on regression and ROC-AUC on classification. Saved results include test metrics, validation–test gaps, selected parameters, trial histories, resource use, timing, and convergence diagnostics. Secondary metrics are MAE for regression and F1, average precision, and log loss for classification. Average precision replaces the earlier generic “PR-AUC” wording. A positive gap means worse test performance: test RMSE minus validation RMSE, or validation AUC minus test AUC.

Preprocessing is fitted inside each training fold: numeric median imputation and standardization, plus categorical most-frequent imputation and one-hot encoding. The shared encoder also applies to LightGBM; native categorical handling remains unimplemented. Current runs use four parallel CV workers and one native thread per model. SVM is limited to datasets with at most 5,000 rows and the Small conditions. Details are in [experiments.md](experiments.md).

### Evaluation and Reporting

[Notebook 03](../notebooks/03_final_analysis.ipynb) now implements the results workflow. It audits all saved files, selects and deduplicates a compatible source cohort for the main comparisons, and analyzes historical source cohorts separately. Comparisons require matching dataset, condition, model, budget, and seed across all four optimizers. Historical files with missing convergence diagnostics remain unknown rather than being assumed converged.

Raw metrics are reported within datasets. Aggregate comparisons rank the optimizers from 1 to 4, average seeds within conditions and conditions within datasets, and combine the four related synthetic generators into one block. The initially proposed best-to-worst 0–1 rescaling was replaced by ranks and explicitly defined relative-effect plots. The notebook exports performance heatmaps, budget gains, runtime ratios, seed variability, resource/time convergence curves, paired Base–Noisy and Base–Small effects, secondary metrics, Pareto tradeoffs, generalization gaps, and convergence sensitivity tables. It also exports every available matched high-budget trajectory as an appendix panel.

Statistical comparisons use dataset blocks rather than treating seeds as independent datasets. Friedman tests with Holm correction across the six model–budget panels and conditional Nemenyi post-hoc comparisons are implemented, with numerical tables instead of the proposed critical-difference diagram. Formal tests are deferred until planned coverage is complete and selected fits have known passing convergence diagnostics. The generated results remain descriptive while those requirements are unmet.

The executed October 30 analysis audited 1,619 files, including 1,618 completed files and one interrupted file across three source versions. The main compatible cohort contains 1,345 deduplicated completed files and 448 complete matched seed–budget comparisons. It exported 13 report figures, 224 trajectory panels, detailed tables, a snapshot manifest, an output index, and results-section evidence under `figures/final_analysis/`. These counts describe that snapshot, not completion of the full design.

### Software and Tools

The implementation uses the following software and tools:

- Python/Jupyter for experimentation and documenting experiments
- NumPy and pandas for data manipulation
- Scikit-learn for loading the OpenML datasets, generating synthetic data, preprocessing, models, metrics, parameter grids and random sampling; custom adapters implement the search loops and successive halving
- LightGBM for gradient-boosted trees
- Optuna for Bayesian optimization
- SciPy for statistical tests
- Matplotlib and Seaborn for result visualization
- Git and GitHub for version control and team collaboration

### Next Steps

The dataset loaders, fold-fitted preprocessing, model factories, four optimizer adapters, experiment runner, result checkpoints, run-level reuse, convergence diagnostics, and final analysis notebook are implemented. The seed target was reduced for compute time; CV folds were not reduced. Remaining work is to address missing compatible comparisons and convergence warnings where feasible, rerun notebook 03 for the final snapshot, and write the results section using practical test-score differences, runtime, robustness, and the documented limitations. The current descriptive results do not support a claim of statistically significant superiority.
