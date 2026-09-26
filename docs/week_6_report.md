# Week 6 Project Report

## 1. Team Information

- **Team name:** Optima Prime
- **Team members:**
  - Danay Fernandes Alfonso
  - Dilek Ozdemirci
  - Ecil Teodoro

## 2. Team Coordination

- **Week 6 meeting date:** Sat, Sep 26 2026
- **Meeting time:** 1pm EDT
- **Communication methods:** Zoom

During the meeting, we discussed the project scope, possible datasets, division of responsibilities, and the implementation approach. We will continue to use Zoom to meet, coordinate development, share results, and track project progress.

## 3. Project Details

### Project Idea

Our main question is whether the best way to tune a model changes when the model, the data, or the available computing budget changes. We plan to test grid search, random search, Bayesian optimization, and successive halving on three models with small, medium, and large search spaces. We will run them on six public datasets and on synthetic data. In the end we hope to identify situations in which a simple search is sufficient and situations in which a more adaptive method is worth the extra complexity.

### Dataset

We will use six public datasets from OpenML together with synthetic datasets generated with Scikit-learn. We picked them so that, as a group, they include regression and classification problems, small and large sample sizes, few and many features, balanced and imbalanced classes, clean and noisy data, and numeric as well as categorical features.

| Dataset | OpenML ID | Task | Rows x Features | Notes |
| --- | --- | --- | --- | --- |
| White Wine Quality | 44971 | Regression | 4,898 x 11 | Small, few features |
| Superconductivity | 44964 | Regression | 21,263 x 81 | Large, nonlinear |
| Communities and Crime | 46286 | Regression | 1,994 x 122 | Small, many features, noisy |
| Bank Marketing | 1461 | Classification | 45,211 x 16 | Large, imbalanced, categorical features |
| Bioresponse | 4134 | Classification | 3,751 x 1,776 | Small, very many features, balanced |
| Default of Credit Card Clients | 42477 | Classification | 30,000 x 23 | Large, imbalanced, noisy labels |
| Synthetic | N/A | Both | 2,000 x 20 | Linear and nonlinear, balanced and imbalanced |

The synthetic data gives us a baseline where we know the true relationship between the predictors and the target. We will also make modified copies of the real datasets. The three large ones will be subsampled down to 2,000 rows, the regression targets will get extra random noise, and 10% of the classification labels will be flipped. The noise only goes into the training data so that the test set still reflects the true signal.

Two datasets changed from our proposal. Madelon turned out to be synthetic (Scikit-learn's make_classification function is based on the same generator), so it would not add anything our own synthetic data does not already cover. We replaced it with Bioresponse, which is real data with a very large number of features. We also dropped Census-Income KDD because, at almost 300,000 rows, tuning it many times would take too long. Default of Credit Card Clients takes its place. More details are in [datasets.md](datasets.md).

### Models

We changed the models as well. Ridge and logistic regression only have one or two hyperparameters, so there is not much for the different search methods to disagree on. We still wanted a simple linear model, so we moved to Elastic Net, and we added two models with more hyperparameters:

| Search space | Model | Hyperparameters tuned |
| --- | --- | --- |
| Small (2-3) | Elastic Net (regression) and logistic regression with elastic-net penalty (classification) | Regularization strength, L1/L2 mix, class weights |
| Medium (3-5) | Support vector machine with RBF kernel | C, gamma, and epsilon or class weights |
| Large (8-10) | LightGBM | Learning rate, number of trees, tree size, sampling rates, regularization |

Going from a small to a large search space should show us whether the size of the search space affects which method works best. LightGBM replaces Scikit-learn's gradient-boosted trees since it trains faster and has more settings to tune. More details are in [models.md](models.md).

### Methodology

Rather than letting one method run longer or try many more settings, every method gets the same search space, the same train/test split, the same cross-validation folds, and the same budget. We split each dataset 80/20 into training and test sets and use 5-fold cross-validation on the training part. We will run each search with a budget of 25 configurations and again with 100. Successive halving is set up so that its total training effort is about the same, even though it looks at more configurations at first. Every run is repeated with 10 different random seeds so we can see how consistent the results are.

We tune for RMSE on the regression problems and ROC-AUC on the classification problems. For each run we will also keep the test score, the gap between the validation and test scores, the total time, the number of configurations tried, and the score and time of every single trial. We will also report MAE for regression, and F1, precision-recall AUC, and log loss for classification.

In total there are 51 combinations of dataset, data version, and model, and each one is tuned by all four methods at both budgets. The SVM is only used on datasets with 5,000 rows or fewer, because its training time grows very fast with the number of rows. The full list is in [experiments.md](experiments.md).

### Evaluation and Reporting

Since the datasets use different metrics, we cannot average the raw scores. Instead we will rank the four methods within each experiment, and we will also rescale the scores so the best method in an experiment gets 0 and the worst gets 1. For the paper we are planning a heatmap that shows which method won on each dataset and model, convergence curves of the best score so far against the number of trials and against time, a critical difference diagram of the average ranks, and box plots of how much the results vary between seeds. All methods share the same splits within a seed, so we will use paired tests (Friedman with a Nemenyi post-hoc test). The detailed results for each dataset will go in an appendix.

### Software and Tools

The project will use the following software and tools:

- Python/Jupyter for experimentation and documenting experiments
- NumPy and pandas for data manipulation
- Scikit-learn for loading the OpenML datasets, generating synthetic data, preprocessing, models, metrics, grid search, random search, and successive halving
- LightGBM for gradient-boosted trees
- Optuna for Bayesian optimization
- SciPy for statistical tests
- Matplotlib and Seaborn for result visualization
- Git and GitHub for version control and team collaboration

### Next Steps

Our next step is to write the code that loads and prepares the datasets, including the subsampled and noisy versions, and to agree on the exact search space for each model. After that we will do a trial run with a single seed for every combination. This will tell us how long the full set of experiments will take and let us test the plotting code early. If it turns out to be too slow, we will cut the number of seeds or folds for the large datasets.
