# Models

The project tunes three models whose search spaces grow from low to high dimension. This makes it possible to test whether the best hyperparameter optimization method changes as the search space gets larger.

## Summary

| Tier | Model | Regression | Classification | Tuned hyperparameters | Successive-halving resource |
| --- | --- | --- | --- | --- | --- |
| Low (2–3) | Elastic Net / elastic-net logistic regression | `sklearn.linear_model.ElasticNet` | `sklearn.linear_model.LogisticRegression(penalty="elasticnet", solver="saga")` | `alpha` (regression) or `C` (classification), `l1_ratio`, `class_weight` (classification) | Training sample size |
| Medium (3–5) | SVM with RBF kernel | `sklearn.svm.SVR` | `sklearn.svm.SVC` | `C`, `gamma`, `epsilon` (regression) or `class_weight` (classification) | Training sample size |
| High (8–10) | LightGBM | `lightgbm.LGBMRegressor` | `lightgbm.LGBMClassifier` | `learning_rate`, `n_estimators`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_split_gain` | Boosting rounds (`n_estimators`) |

## Why each model is included

### Low tier: Elastic Net / elastic-net logistic regression

A convex linear model with a small, smooth search space. Grid search should be competitive here, which makes it the baseline for the other optimizers. Low dimension does not mean low tunability: regularization strength still has a large effect on performance (Probst, Boulesteix & Bischl, 2019).

### Medium tier: SVM with RBF kernel

A few continuous hyperparameters that interact strongly. The good region of `C` × `gamma` is a narrow diagonal band on a log scale, which a coarse grid can miss but random search and Bayesian optimization can find. **Limitation:** training cost grows at least with the square of the number of rows, so the SVM runs only on the small datasets and on subsampled versions of the large ones.

### High tier: LightGBM

A gradient-boosted tree model with many hyperparameters, where only a few have a large effect. Under a shared budget, grid search can only cover a few of them and must fix the rest at their defaults. This is the setting where random search, Bayesian optimization, and successive halving are expected to do better. The number of boosting rounds is a natural resource for successive halving.

## Notes

- All numeric features are standardized for Elastic Net and SVM. Categorical features are one-hot encoded for Elastic Net and SVM, and passed as native categoricals to LightGBM.
- The search space for each model is shared by all four optimizers. The exact ranges and scales will be defined in a separate search-space specification.
