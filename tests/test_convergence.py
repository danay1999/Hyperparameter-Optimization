import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from hpo_project.models import make_model
from hpo_project.optimizers import optimize
from hpo_project.convergence import fit_with_diagnostics


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=90, n_features=15, random_state=1)
        self.X, self.y = pd.DataFrame(X), pd.Series(y)
        self.estimator = ElasticNet(alpha=0.0001, max_iter=1, tol=1e-12)

    def test_model_iteration_limit(self):
        self.assertEqual(make_model('elastic_net', 'regression', 1, 1).get_params()['model__max_iter'], 50000)

    def test_refit_diagnostics(self):
        _, messages = fit_with_diagnostics(self.estimator, self.X, self.y)
        self.assertTrue(messages)

    def test_all_adapters_capture_warnings(self):
        folds = list(KFold(3).split(self.X))
        for optimizer in ['grid', 'random', 'bayesian', 'halving']:
            with self.subTest(optimizer=optimizer):
                trials = optimize(self.estimator, {'alpha': {'choices': [0.0001]}}, optimizer, 2,
                    self.X, self.y, folds, 'neg_root_mean_squared_error', 1, 1)
                self.assertTrue(any(not t['converged'] and t['convergence_warnings'] for t in trials))

    def test_parallel_cv_diagnostics(self):
        trials = optimize(self.estimator, {'alpha': {'choices': [0.0001]}}, 'grid', 1,
            self.X, self.y, list(KFold(3).split(self.X)), 'neg_root_mean_squared_error', 1, 2)
        self.assertEqual(len(trials[0]['convergence_warnings']), 3)
