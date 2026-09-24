# Week 6 Project Report

## 1. Team Information

- **Team name:** Optima Prime
- **Team members:**
  - Danay Fernandes Alfonso
  - Dilek Ozdemirci
  - Ecil Teodoro

## 2. Team Coordination

- **Week 8 meeting date:** Sun, Sep 27 2026
- **Meeting time:** 6pm EDT
- **Communication methods:** Zoom

During the meeting, we discussed the project scope, possible datasets, division of responsibilities, and the implementation approach. We will continue to use Zoom to meet, coordinate development, share results, and track project progress.

## 3. Project Details

### Dataset

The project will initially use controlled synthetic regression and classification datasets generated with Scikit-learn. These datasets will allow the team to systematically vary characteristics such as sample size, noise level, dimensionality, and the complexity of the relationship between the predictors and target. One or more public real-world datasets may also be added later to validate whether the findings extend beyond synthetic data.

### Project Idea and Methodology

Our main question is whether the best way to tune a model changes when the model or the data changes. We plan to test grid search, random search, Bayesian optimization, and successive halving on ridge regression, logistic regression, and gradient-boosted trees. Rather than letting one method run longer or try many more settings, we will set the same limit on the number of trials or the available run time for each one.

We will then compare the best predictive score each method finds, how long the search takes, and how quickly it reaches a strong result. Repeating the experiments will also show whether the results are consistent or depend heavily on a particular run. By changing the sample size, noise level, and number of features in the synthetic datasets, we hope to identify situations in which a simple search is sufficient and situations in which a more adaptive method is worth the extra complexity.

### Software and Tools

The project will use the following software and tools:

- Python/Jupyter for exploratory analysis, experimentation, and documenting experiments
- NumPy and pandas for data manipulation
- Scikit-learn for dataset generation, preprocessing, models, metrics, and baseline search methods
- Optuna for Bayesian optimization and pruning or successive-halving methods
- Matplotlib and Seaborn for result visualization
- Git and GitHub for version control and team collaboration
