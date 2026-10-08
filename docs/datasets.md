# Datasets

The project uses six public datasets from OpenML and one family of synthetic datasets. Together they cover regression and binary classification, small and large samples, low- and high-dimensional feature spaces, linear and nonlinear relationships, balanced and imbalanced classes, clean and noisy data, and numeric and categorical features.

## Summary

| # | Dataset | OpenML ID | Task | Rows × features | Size | Dimension | Balanced | Noisy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | White Wine Quality | [44971](https://www.openml.org/d/44971) | Regression | 4,898 × 11 | Small | Low | N/A | Moderate |
| 2 | Superconductivity | [44964](https://www.openml.org/d/44964) | Regression | 21,263 × 81 | Large | Medium | N/A | Clean |
| 3 | Communities and Crime | [46286](https://www.openml.org/d/46286) | Regression | 1,994 × 122 | Small | Medium–High | N/A | Noisy |
| 4 | Bank Marketing | [1461](https://www.openml.org/d/1461) | Binary classification | 45,211 × 16 | Large | Low | No (11.7% positive) | Clean |
| 5 | Bioresponse | [4134](https://www.openml.org/d/4134) | Binary classification | 3,751 × 1,776 | Small | High | Yes (45.8% positive) | Moderate |
| 6 | Default of Credit Card Clients | [42477](https://www.openml.org/d/42477) | Binary classification | 30,000 × 23 | Large | Low | No (22.1% positive) | Noisy |
| 7 | Synthetic (scikit-learn) | N/A | Both | Configurable | Configurable | Configurable | Configurable | Configurable |

Feature counts exclude the target. Categories use these thresholds:

- **Size:** small < 5,000 rows; large ≥ 20,000 rows.
- **Dimension:** low < 30 features; medium 30–200; high > 200.
- **Noisy:** how noisy the original data is before any controlled noise is added. "Noisy" means the best achievable score is well below perfect because of label noise, measurement error, or heavy missing data.

## Descriptions

### 1. White Wine Quality

Physicochemical measurements (acidity, residual sugar, sulphates, alcohol, and so on) of Portuguese *vinho verde* white wines. The target is a sensory quality score from 0 to 10 given by wine tasters. All features are numeric with no missing values. The target is an integer and comes from subjective ratings, so it carries moderate noise. The relationship is mildly nonlinear: tree ensembles beat linear models, but not by much. This dataset is part of the OpenML-CTR23 regression benchmark.

### 2. Superconductivity

81 features derived from the chemical formulas of 21,263 superconductors, such as means, weighted means, and ranges of atomic mass, radius, valence, and thermal conductivity. The target is the critical temperature (K). All features are numeric with no missing values. The relationship is strongly nonlinear, and gradient-boosted trees gain a lot from tuning. This dataset is part of the OpenML-CTR23 regression benchmark.

### 3. Communities and Crime

Socio-economic data from the 1990 US Census, law-enforcement data from the 1990 LEMAS survey, and crime data from the 1995 FBI Uniform Crime Report for 1,994 US communities. The target is the normalized per-capita violent crime rate. About 25 attributes are roughly 84% missing and should be dropped, along with non-predictive identifiers (state, county, community name, fold), which leaves about 99 numeric features. It is small, relatively high-dimensional, and naturally noisy, and regularized linear models do well on it, so it is the closest to linear of the real datasets.

### 4. Bank Marketing

Records from direct phone-marketing campaigns run by a Portuguese bank (the `bank-full` version). The target is whether the client subscribed to a term deposit. It has 10 categorical features (job, marital status, education, contact type, month, and so on) and 6 numeric features. Classes are imbalanced (11.7% "yes"). **The `duration` column (V12 in the OpenML version) must be dropped**: it is only known after the call has ended, so it leaks the target.

### 5. Bioresponse

Molecular descriptors for 3,751 molecules from the Boehringer Ingelheim Kaggle competition. The target is whether a molecule produced a biological response. All 1,776 features are numeric and pre-normalized, and the classes are close to balanced. It is the high-dimensional classification case: there are about two samples per feature, so regularization strength matters a lot. It replaces Madelon, which is synthetic. Scikit-learn's `make_classification` uses the algorithm that generated Madelon, so the synthetic datasets already cover it.

### 6. Default of Credit Card Clients

Demographic, credit-limit, repayment-history, bill, and payment data for 30,000 credit-card clients in Taiwan (April–September 2005). The target is default on the next month's payment (22.1% positive). Most features are numeric. Sex, education, marital status, and repayment status are integer-coded categories. The label is noisy: even well-tuned models reach only moderate ROC-AUC, which makes it a realistic noisy-label case. It replaces Census-Income KDD, which is about ten times larger, has a survey-weight column that must not be used as a feature, and contains many duplicate or conflicting rows.

### 7. Synthetic datasets

The study includes four synthetic datasets generated with scikit-learn. Each contains **2,000 observations and 20 numerical features**, with generation controlled by the repetition seed. They provide controlled settings for comparing optimizers under linearity, nonlinearity, and class imbalance.

| Dataset | Task | Generation settings | Purpose |
|---|---|---|---|
| **S1: Linear regression** | Regression | `make_regression`; 10 informative features, 10 non-informative features, Gaussian target noise with standard deviation 10 | Evaluate optimization when the underlying relationship is linear. |
| **S2: Friedman regression** | Regression | `make_friedman1`; 5 informative features, 15 non-informative features, Gaussian target noise with standard deviation 1 | Evaluate optimization with nonlinear relationships and feature interactions. |
| **S3: Balanced classification** | Binary classification | `make_classification`; 5 informative, 5 redundant, and 10 non-informative features; approximately equal class proportions | Provide a classification baseline with limited label noise. |
| **S4: Imbalanced classification** | Binary classification | Same feature structure as S3, with nominal class proportions of 90% and 10% | Examine optimizer performance under class imbalance. |

For both classification datasets, `flip_y=0.01` randomly reassigns approximately 1% of labels; this does not necessarily change every selected label. Consequently, realized class proportions may differ slightly from the nominal settings.

Unlike the real datasets, the synthetic datasets are evaluated only in the **base condition**. Their generator noise is present before the train/test split and therefore affects both partitions. Each repetition generates a new dataset, then uses an 80/20 train/test split, stratified for classification. Within a repetition, every optimizer receives the same generated data, split, and cross-validation folds.

## Controlled variants

Controlled versions of the real datasets isolate specific conditions:

- **Sample size:** stratified subsampling of the large datasets (2, 4, 6) to create small versions.
- **Regression noise:** Gaussian noise added to the features or the target of datasets 1–3.
- **Classification noise:** random label flipping at fixed rates in datasets 4–6.

## Analysis

The study will include dataset analysis, explaining why the datasets provide different optimization challenges and quantofying how reliably the optimizers differ.

| Analysis | What to include | Why it matters |
| --- | --- | --- |
| Dataset characteristics | Sample size, number of numerical/categorical features, missingness, feature-to-sample ratio | Explains differences in training cost and model behavior |
| Target distribution | Regression: median, spread, skewness, outliers. Classification: class counts and proportions | Identifies difficult targets and class imbalance |
| Feature relationships | Selected correlations, redundant features, categorical cardinality | Provides context for Elastic Net, SVM, and LightGBM behavior |
| Experimental conditions | How sample size and training-label distributions change under `small` and `noisy` conditions | Demonstrates what the interventions actually changed |
| Optimizer performance | Mean and standard deviation across seeds, paired score differences, uncertainty intervals | Shows the size and stability of performance differences |
| Computational efficiency | Matched runtime ratios, budget gains, score-versus-time plots | Shows whether improved accuracy justifies additional computation |

The study will include one compact summary table and a few representative plots. Large correlation matrices for every dataset would overwhelm the report; so we have to put detailed diagnostics in an appendix.

## Loading

```python
from sklearn.datasets import fetch_openml

DATASETS = {
    "white_wine": 44971,
    "superconductivity": 44964,
    "communities_crime": 46286,
    "bank_marketing": 1461,
    "bioresponse": 4134,
    "credit_default": 42477,
}

X, y = fetch_openml(data_id=DATASETS["white_wine"], as_frame=True, return_X_y=True)
```

## Sources

- [OpenML-CTR23 regression benchmark suite (study 353)](https://www.openml.org/search?type=study&study_type=task&id=353)
- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [UCI Census-Income KDD documentation](https://kdd.ics.uci.edu/databases/census-income/census-income.data.html)
- [scikit-learn `make_classification` (Madelon note)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
