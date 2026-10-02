import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
import yaml
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, roc_auc_score, f1_score, average_precision_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from .datasets import load_dataset, prepare_split
from .models import make_model
from .convergence import fit_with_diagnostics
from .optimizers import optimize


def matrix(config):
    for dataset, spec in config['datasets'].items():
        for condition in spec['conditions']:
            for model in ['elastic_net', 'svm', 'lightgbm']:
                if model == 'svm' and not spec['small'] and condition != 'small':
                    continue
                for optimizer in config['optimizers']:
                    budgets = [max(config['budgets'])] if optimizer in ['random', 'bayesian'] else config['budgets']
                    for budget in budgets:
                        for seed in config['seeds']:
                            yield dict(dataset=dataset, condition=condition, model=model, optimizer=optimizer, budget=budget, seed=seed)


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    temp.replace(path)


def run(config, spaces, selection, output='results/runs'):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for item in matrix(config):
        if any(value is not None and item[key] != value for key, value in selection.items()):
            continue
        spec = config['datasets'][item['dataset']]
        space_spec = spaces[item['model']]
        space = {f'model__{k}': v for k, v in {**space_spec.get('common', {}), **space_spec.get(spec['task'], {})}.items()}
        identity = {**item, 'dataset_spec': spec, 'space': space, 'cv_folds': config['cv_folds'], 'n_jobs': config['n_jobs'], 'protocol_version': 1}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        identity['code_revision'] = revision
        identity['source_hash'] = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('src/hpo_project').rglob('*.py')))).hexdigest()
        run_id = '-'.join(str(v) for v in item.values()) + '-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        path = root / f'{run_id}.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
            print(f'Skip {run_id}', flush=True)
            continue
        result = dict(run_id=run_id, config=identity, status='running', versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'optuna', 'lightgbm']})
        save(path, result)
        print(f'Run {run_id}', flush=True)
        try:
            X, y = load_dataset(item['dataset'], spec, item['seed'])
            Xtr, Xte, ytr, yte = prepare_split(X, y, spec['task'], item['condition'], item['seed'])
            cv_class = KFold if spec['task'] == 'regression' else StratifiedKFold
            folds = list(cv_class(config['cv_folds'], shuffle=True, random_state=item['seed']).split(Xtr, ytr))
            estimator = make_model(item['model'], spec['task'], item['seed'], config['n_jobs'])
            scoring = 'neg_root_mean_squared_error' if spec['task'] == 'regression' else 'roc_auc'
            trials = optimize(estimator, space, item['optimizer'], item['budget'], Xtr, ytr, folds, scoring, item['seed'], config['n_jobs'])
            result['trials'] = trials
            summaries = []
            budgets = config['budgets'] if item['optimizer'] in ['random', 'bayesian'] else [item['budget']]
            for budget in budgets:
                eligible = [t for t in trials if t['resource'] == 1.0] if item['optimizer'] == 'halving' else trials[:budget]
                best = max(eligible, key=lambda t: t['score'])
                fitted, refit_warnings = fit_with_diagnostics(clone(estimator).set_params(**best['params']), Xtr, ytr)
                if refit_warnings:
                    print(f'Convergence warning during refit for budget {budget}; diagnostics saved.', flush=True)
                pred = fitted.predict(Xte)
                if spec['task'] == 'regression':
                    metrics = dict(rmse=root_mean_squared_error(yte, pred), mae=mean_absolute_error(yte, pred))
                    validation, test = -best['score'], metrics['rmse']
                    gap = test - validation
                else:
                    probability = fitted.predict_proba(Xte)[:, 1]
                    metrics = dict(roc_auc=roc_auc_score(yte, probability), f1=f1_score(yte, pred), average_precision=average_precision_score(yte, probability), log_loss=log_loss(yte, probability))
                    validation, test = best['score'], metrics['roc_auc']
                    gap = validation - test
                summaries.append(dict(budget=budget, best_params=best['params'], validation_score=validation, test_score=test,
                    generalization_gap=gap, metrics=metrics, best_trial_converged=best['converged'], refit_converged=not refit_warnings, refit_convergence_warnings=refit_warnings, configurations=len(trials) if item['optimizer'] == 'halving' else len(trials[:budget]), full_fidelity_equivalents=sum(t['resource'] for t in trials) if item['optimizer'] == 'halving' else len(trials[:budget]), optimization_seconds=trials[-1]['elapsed_seconds'] if item['optimizer'] == 'halving' else trials[min(budget, len(trials))-1]['elapsed_seconds']))
            result.update(status='complete', summaries=summaries)
        except Exception:
            result.update(status='failed', error=traceback.format_exc())
            save(path, result)
            raise
        save(path, result)
