"""Capture convergence diagnostics in the process performing each fit."""
import warnings
from sklearn.exceptions import ConvergenceWarning


def fit_with_diagnostics(estimator, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        fitted = estimator.fit(X, y)
    diagnostics = []
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            diagnostics.append(str(warning.message))
        else:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return fitted, diagnostics
