import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
import yaml
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, roc_auc_score, f1_score, average_precision_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from .datasets import load_dataset, prepare_split
from .models import make_model
from .convergence import fit_with_diagnostics
from .optimizers import optimize
from .resume import index_results, find_completed, model_settings, combination


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


def run(config, spaces, selection, output='results/runs', force=False, accept_legacy=False, dry_run=False):
    root = Path(output)
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
    completed = index_results(root)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'optuna', 'lightgbm']}
    cv_jobs = config.get('cv_jobs', config.get('n_jobs', 1))
    model_threads = config.get('model_threads', 1)
    if not isinstance(cv_jobs, int) or cv_jobs < 1 or not isinstance(model_threads, int) or model_threads < 1:
        raise ValueError('cv_jobs and model_threads must be positive integers')
    counts = {'planned': 0, 'skip': 0, 'run': 0}
    for item in matrix(config):
        if any(value is not None and item[key] != value for key, value in selection.items()):
            continue
        spec = config['datasets'][item['dataset']]
        space_spec = spaces[item['model']]
        space = {f'model__{k}': v for k, v in {**space_spec.get('common', {}), **space_spec.get(spec['task'], {})}.items()}
        estimator = make_model(item['model'], spec['task'], item['seed'], model_threads)
        budgets = config['budgets'] if item['optimizer'] in ['random', 'bayesian'] else [item['budget']]
        identity = {**item, 'dataset_spec': spec, 'space': space, 'cv_folds': config['cv_folds'], 'n_jobs': cv_jobs, 'cv_jobs': cv_jobs, 'model_threads': model_threads, 'model_settings': model_settings(estimator), 'protocol_version': 1}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        identity['code_revision'] = revision
        identity['source_hash'] = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('src/hpo_project').rglob('*.py')))).hexdigest()
        run_id = '-'.join(str(v) for v in item.values()) + '-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        path = root / f'{run_id}.json'
        counts['planned'] += 1
        match, rejected = find_completed(completed, identity, versions, budgets, accept_legacy)
        if match and not force:
            counts['skip'] += 1
            print(f"Skip {match[0].name}: {match[2]}", flush=True)
            continue
        counts['run'] += 1
        for filename, reason in rejected:
            print(f"Cannot reuse {filename}: {reason}", flush=True)
        if dry_run:
            print(f"Would run {run_id}" + (' (forced)' if force else ''), flush=True)
            continue
        # --force preserves previous files, including when the exact ID matches.
        if path.exists() and force:
            import time
            path = root / f'{run_id}-forced-{time.time_ns()}.json'
            run_id = path.stem
        result = dict(run_id=run_id, config=identity, status='running', versions=versions)
        save(path, result)
        print(f'Run {run_id}', flush=True)
        try:
            X, y = load_dataset(item['dataset'], spec, item['seed'])
            Xtr, Xte, ytr, yte = prepare_split(X, y, spec['task'], item['condition'], item['seed'])
            cv_class = KFold if spec['task'] == 'regression' else StratifiedKFold
            folds = list(cv_class(config['cv_folds'], shuffle=True, random_state=item['seed']).split(Xtr, ytr))
            scoring = 'neg_root_mean_squared_error' if spec['task'] == 'regression' else 'roc_auc'
            def checkpoint(trials):
                result['trials'] = trials
                save(path, result)
                last = trials[-1]
                print(f"  Trial {len(trials)}: CV score={last['score']:.6g}, resource={last['resource']:.3g}, elapsed={last['elapsed_seconds']/60:.1f} min", flush=True)
            trials = optimize(estimator, space, item['optimizer'], item['budget'], Xtr, ytr, folds, scoring, item['seed'], cv_jobs, model_threads=model_threads, progress=checkpoint)
            result['trials'] = trials
            summaries = []
            budgets = config['budgets'] if item['optimizer'] in ['random', 'bayesian'] else [item['budget']]
            for budget in budgets:
                eligible = [t for t in trials if t['resource'] == 1.0] if item['optimizer'] == 'halving' else trials[:budget]
                best = max(eligible, key=lambda t: t['score'])
                with threadpool_limits(limits=model_threads):
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
        except KeyboardInterrupt:
            result.update(status='interrupted')
            save(path, result)
            raise
        except Exception:
            result.update(status='failed', error=traceback.format_exc())
            save(path, result)
            raise
        save(path, result)
    print(f"Planned: {counts['planned']}; skipped: {counts['skip']}; {'remaining' if dry_run else 'executed'}: {counts['run']}", flush=True)
    return counts
