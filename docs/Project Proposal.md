# Project Proposal

*Updated October 30, 2026 to reflect the implemented study.*

## Project Goal

We compare grid search, random search, Bayesian optimization, and successive halving to determine how their predictive performance and computational cost depend on the model, data characteristics, and search budget. The practical question is when a simple search is sufficient and when an adaptive method provides enough improvement to justify its cost.

## Data and Models

The implemented benchmark contains six real OpenML datasets: White Wine Quality, Superconductivity, Communities and Crime, Bank Marketing, Bioresponse, and Default of Credit Card Clients. Four synthetic generators provide linear and nonlinear regression and balanced and imbalanced classification tasks. Dataset sources and cleaning rules are documented in [datasets.md](datasets.md).

Real datasets have Base and Noisy conditions. The three large datasets also have a Small condition containing 2,000 training observations. We first reserve a clean 20% test set, then subsample or corrupt only the training data. Regression noise has standard deviation 0.5 times the training target's standard deviation; classification noise flips 10% of training labels. The held-out test set remains shared across conditions for a dataset and seed. Synthetic datasets use the Base condition only.

We tune Elastic Net regression or elastic-net logistic regression, an RBF support vector machine, and LightGBM. SVM is restricted to small datasets and Small conditions. This produces 51 dataset–condition–model cells. The search spaces contain two or three parameters for the linear models, three for SVM, and ten for LightGBM. For halving, LightGBM's tree count is assigned by the resource schedule. Model-family comparisons do not isolate the causal effect of search-space dimension.

All models use preprocessing fitted within each CV training fold: median imputation and standardization for numeric features, and most-frequent imputation and one-hot encoding for categorical features. Native LightGBM categorical handling was considered but is not implemented.

## Methodology

The initial idea of an equivalent time budget was refined into budgets of **25 and 100 full-fidelity cross-validation evaluations**. Actual elapsed optimization time is recorded separately; equal resource budgets do not guarantee equal wall-clock time.

For each repetition, all optimizers share an 80/20 train/test split and the same five cross-validation folds, stratified for classification. **We reduced the original plan of ten seeds to five seeds (1–5) for all cells because of compute time.** Five-fold CV and both budget levels were retained. The five-seed design targets 1,530 optimization runs and 2,040 optimizer–budget–seed summaries; these are planned totals, not counts of completed compatible runs.

Random and Bayesian search run once to budget 100; the first 25 trials supply the low-budget result, with a separate refit at each budget. Grid and halving run separately at each budget. Grid search builds a rectangular grid with at most the requested number of configurations. Bayesian optimization uses Optuna's TPE sampler. The custom successive-halving adapter uses three resource fractions (1/9, 1/3, and 1), retains approximately one third of candidates at each rung, and keeps validation folds fixed. Its resource is training-fold sample size for linear models and SVM, and boosting rounds for LightGBM. Only full-resource candidates can supply its final selected configuration.

Tuning maximizes ROC-AUC for classification and minimizes RMSE for regression. The best configuration is refit on the training set and evaluated on the held-out test set. Secondary metrics are MAE, F1, average precision, and log loss. Average precision is the implemented PR metric. Each trial records parameters, validation score, resource use, timing, and convergence warnings. The current execution setting uses four parallel CV workers and one native model thread per worker.

## Analysis and Reporting

[Notebook 03](../notebooks/03_final_analysis.ipynb) audits every saved run and exports figures, CSV tables, provenance, and evidence for the results section. Main comparisons use matched optimizer results from one compatible source cohort; historical cohorts are analyzed separately. Duplicates and unfinished runs remain visible in the audit.

The analysis covers test performance and optimizer ranks, gains from budget 25 to 100, runtime, seed variability, noise and sample-size robustness, secondary metrics, quality–time tradeoffs and Pareto efficiency, generalization gaps, convergence sensitivity, and best-so-far validation trajectories against resource and time. Raw scores and effects are reported within datasets; aggregate comparisons use ranks. Seeds are averaged within conditions, conditions within datasets, and the four related synthetic generators form one block.

Friedman tests with Holm correction across model–budget panels and conditional Nemenyi post-hoc comparisons are implemented. They remain deferred until planned coverage and convergence requirements pass. The current saved results are therefore descriptive, with incomplete coverage and solver warnings disclosed. Absence of a significant result would not establish equivalence.

## Expected Outcome and Remaining Work

The deliverable is an evidence-based comparison of which methods perform well, how much improvement they provide, what that improvement costs, and whether patterns persist across models, budgets, and data conditions. The experiment runner and report-analysis workflow are built. Remaining work is to resolve coverage and convergence limitations where feasible, refresh the final analysis snapshot, and write the results with practical effects and uncertainty alongside ranks. The detailed protocol is in [experiments.md](experiments.md), and sources are listed in [references.md](references.md).
