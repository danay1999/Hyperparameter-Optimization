from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_regression, make_friedman1, make_classification
from sklearn.model_selection import train_test_split


def load_dataset(name, spec, seed):
    if name.startswith("synthetic_"):
        kwargs = dict(n_samples=2000, n_features=20, random_state=seed)
        if name == "synthetic_linear":
            X, y = make_regression(**kwargs, n_informative=10, noise=10)
        elif name == "synthetic_friedman":
            X, y = make_friedman1(**kwargs, noise=1)
        else:
            X, y = make_classification(**kwargs, n_informative=5, n_redundant=5,
                weights=[0.9, 0.1] if name.endswith("imbalanced") else None, flip_y=0.01)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), pd.Series(y)
    X, y = fetch_openml(data_id=spec["id"], as_frame=True, return_X_y=True,
                        data_home=str(Path("data/openml")))
    if name == "bank_marketing":
        X = X.drop(columns=[c for c in X if c.lower() in {"duration", "v12"}])
    if name == "communities_crime":
        X = X.replace("?", np.nan)
        X = X.drop(columns=[c for c in X if c.lower() in {"state", "county", "community", "communityname", "fold"}])
        X = X.apply(pd.to_numeric, errors="coerce")
        # Fixed dataset-specific exclusion; do not select columns using held-out data.
        excluded = {"otherpercap", "lemasswornft", "lemasftperpop", "lemasftofficfield", "lemasftofficops", "lemasftofficpatrol", "lemasftofficother", "lemasftofficperc", "lemaspctofficfield", "lemaspctofficops", "lemaspctofficpatrol", "lemaspctofficother", "numkindsdrugsseiz", "policreqperoffic", "policperpop", "racialmatchcommpol", "pctpolicwhite", "pctpolicblack", "pctpolichisp", "pctpolicasian", "pctpolicminor", "officassgndrugunits", "numkindsdrugsseiz", "policaveotworked", "policcars", "policoperbudg", "lemaspctpoliconpatr", "lemasgangunitdeploy", "policbudgperpop"}
        X = X.drop(columns=[c for c in X if c.lower() in excluded])
    if name == "credit_default":
        X = X.drop(columns=[c for c in X if c.lower() == "id"])
        for c in X:
            if c.lower() in {"sex", "education", "marriage"} or c.lower().startswith("pay_") and c.lower() not in {f"pay_amt{i}" for i in range(1, 7)}:
                X[c] = X[c].astype("category")
    if spec["task"] == "classification":
        labels = sorted(y.astype(str).unique())
        if len(labels) != 2:
            raise ValueError("Only binary classification is supported")
        y = y.astype(str).map({labels[0]: 0, labels[1]: 1})
    else:
        y = pd.to_numeric(y)
    return X, y


def prepare_split(X, y, task, condition, seed):
    # Hold out clean test data before subsampling or corrupting training labels.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                       stratify=y if task == "classification" else None)
    if condition == "small" and len(Xtr) > 2000:
        strata = ytr if task == "classification" else pd.qcut(ytr, 10, duplicates="drop")
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=2000, random_state=seed, stratify=strata)
    if condition == "noisy":
        rng = np.random.default_rng(seed)
        ytr = ytr.copy()
        if task == "regression":
            ytr += rng.normal(0, 0.5 * ytr.std(), len(ytr))
        else:
            idx = rng.choice(len(ytr), int(round(0.1 * len(ytr))), replace=False)
            ytr.iloc[idx] = 1 - ytr.iloc[idx]
    return Xtr, Xte, ytr, yte
