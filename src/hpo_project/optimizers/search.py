import time
import sys
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from ..convergence import fit_with_diagnostics
import numpy as np
from scipy.stats import loguniform, uniform, randint
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from sklearn.metrics import get_scorer


def optimize(estimator, space, optimizer, budget, X, y, folds, scoring, seed, n_jobs, model_threads=1, progress=None):
    distributions = {}
    for key, spec in space.items():
        if 'choices' in spec:
            distributions[key] = spec['choices']
        elif spec.get('type') == 'int':
            distributions[key] = randint(spec['low'], spec['high'] + 1)
        elif spec.get('scale') == 'log':
            distributions[key] = loguniform(spec['low'], spec['high'])
        else:
            distributions[key] = uniform(spec['low'], spec['high'] - spec['low'])
    trials = []
    start = time.perf_counter()

    def fit_fold(params, train, valid):
        with threadpool_limits(limits=model_threads):
            fitted, diagnostics = fit_with_diagnostics(clone(estimator).set_params(**params), X.iloc[train], y.iloc[train])
        score = get_scorer(scoring)(fitted, X.iloc[valid], y.iloc[valid])
        return float(score), diagnostics

    def report(diagnostics):
        if diagnostics:
            print(f"Convergence warning: {len(diagnostics)} fold fit(s) did not converge; diagnostics saved with trial.", file=sys.stderr, flush=True)

    def evaluate_folds(params, trial_folds):
        with parallel_config(backend='loky', inner_max_num_threads=model_threads):
            return Parallel(n_jobs=n_jobs)(delayed(fit_fold)(params, train, valid) for train, valid in trial_folds)

    def evaluate(params):
        tick = time.perf_counter()
        outcomes = evaluate_folds(params, folds)
        score = float(np.mean([value for value, _ in outcomes]))
        diagnostics = [message for _, messages in outcomes for message in messages]
        report(diagnostics)
        trials.append(dict(params=params, score=score, seconds=time.perf_counter()-tick,
                           elapsed_seconds=time.perf_counter()-start, resource=1.0, converged=not diagnostics, convergence_warnings=diagnostics))
        if progress:
            progress(trials)
        return score

    if optimizer == 'bayesian':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        def objective(trial):
            params = {}
            for key, spec in space.items():
                if 'choices' in spec:
                    params[key] = trial.suggest_categorical(key, spec['choices'])
                elif spec.get('type') == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'])
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=spec.get('scale') == 'log')
            return evaluate(params)
        study.optimize(objective, n_trials=budget)
    elif optimizer == 'grid':
        # Grow a rectangular grid without exceeding the evaluation budget.
        counts = {key: 1 for key in space}
        while True:
            candidates = [k for k, s in space.items() if ('choices' not in s or counts[k] < len(s['choices']))
                and np.prod([v + (k == key) for key, v in counts.items()]) <= budget]
            if not candidates:
                break
            counts[min(candidates, key=lambda k: counts[k])] += 1
        grid = {}
        for key, spec in space.items():
            n = counts[key]
            if 'choices' in spec:
                values = spec['choices'][:n]
            else:
                values = (np.geomspace if spec.get('scale') == 'log' else np.linspace)(spec['low'], spec['high'], n)
                values = list(dict.fromkeys(int(round(v)) if spec.get('type') == 'int' else float(v) for v in values))
            grid[key] = values
        for params in ParameterGrid(grid):
            evaluate(params)
    elif optimizer == 'random':
        for params in ParameterSampler(distributions, n_iter=budget, random_state=seed):
            evaluate(params)
    elif optimizer == 'halving':
        # Three rungs, nested training subsets, unchanged validation folds.
        # Budget counts training-resource fractions; timing measures actual work.
        fractions = [1/9, 1/3, 1.0]
        count = 1
        def cost(n):
            total = 0
            for fraction in fractions:
                total += n * fraction
                n = max(1, int(np.ceil(n / 3)))
            return total
        while cost(count + 1) <= budget:
            count += 1
        tree_resource = 'model__n_estimators' in space
        candidate_space = {k: v for k, v in distributions.items() if not (tree_resource and k == 'model__n_estimators')}
        candidates = list(ParameterSampler(candidate_space, n_iter=count, random_state=seed))
        full_trees = space.get('model__n_estimators', {}).get('high', 1)
        for rung, fraction in enumerate(fractions):
            scored = []
            for params in candidates:
                params = dict(params)
                if tree_resource:
                    params['model__n_estimators'] = max(1, int(full_trees * fraction))
                tick = time.perf_counter()
                trial_folds = []
                for train, valid in folds:
                    if not tree_resource and fraction < 1:
                        labels = y.iloc[train] if scoring == 'roc_auc' else None
                        train, _ = train_test_split(train, train_size=max(2, int(len(train)*fraction)),
                            random_state=seed, stratify=labels)
                    trial_folds.append((train, valid))
                outcomes = evaluate_folds(params, trial_folds)
                diagnostics = [message for _, messages in outcomes for message in messages]
                score = float(np.mean([value for value, _ in outcomes]))
                report(diagnostics)
                trials.append(dict(params=params, score=score, seconds=time.perf_counter()-tick,
                    elapsed_seconds=time.perf_counter()-start, resource=fraction, rung=rung, converged=not diagnostics, convergence_warnings=diagnostics))
                if progress:
                    progress(trials)
                scored.append((score, params))
            candidates = [params for _, params in sorted(scored, key=lambda pair: pair[0], reverse=True)[:max(1, int(np.ceil(len(scored)/3)))]]
    else:
        raise ValueError(optimizer)
    return trials
