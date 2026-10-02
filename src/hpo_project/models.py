from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from .preprocessing import make_preprocessor


def make_model(name, task, seed, n_jobs):
    regression = task == "regression"
    if name == "elastic_net":
        model = ElasticNet(max_iter=50000, random_state=seed) if regression else LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000, random_state=seed, n_jobs=n_jobs)
    elif name == "svm":
        model = SVR() if regression else SVC(probability=True, random_state=seed)
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor, LGBMClassifier
        model = (LGBMRegressor if regression else LGBMClassifier)(random_state=seed, n_jobs=n_jobs,
            verbosity=-1, subsample_freq=1)
    else:
        raise ValueError(name)
    # Pilot uses one shared fold-fitted encoder; native LightGBM categoricals are pending.
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])
