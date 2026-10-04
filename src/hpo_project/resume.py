"""Experiment identity and explicit compatibility checks for saved results."""
import json
from collections import defaultdict
from pathlib import Path

KEY_FIELDS = ('dataset', 'condition', 'model', 'optimizer', 'budget', 'seed')


def combination(config):
    return tuple(config.get(key) for key in KEY_FIELDS)


def model_settings(estimator):
    model = estimator['model']
    params = model.get_params(deep=False).copy()
    # Parallelism is execution metadata. Normalize the equivalent sklearn APIs.
    params.pop('n_jobs', None)
    if type(model).__name__ == 'LogisticRegression':
        params.pop('penalty', None)
        params['regularization'] = 'elastic_net_via_C_and_l1_ratio'
    return {'class': f'{type(model).__module__}.{type(model).__name__}', 'params': params,
            'preprocessing_version': 1}


def index_results(directory):
    index = defaultdict(list)
    for path in sorted(Path(directory).glob('*.json')):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f'Ignore unreadable result {path.name}: {error}', flush=True)
            continue
        if result.get('status') == 'complete':
            index[combination(result.get('config', {}))].append((path, result))
    return index


def compatible(saved, identity, versions, budgets, accept_legacy=False):
    old = saved.get('config', {})
    for key in ('dataset_spec', 'space', 'cv_folds', 'protocol_version'):
        if old.get(key) != identity.get(key):
            return False, f'{key} differs'
    if not set(budgets).issubset({s.get('budget') for s in saved.get('summaries', [])}):
        return False, 'requested budget summaries missing'
    if not saved.get('trials'):
        return False, 'trial history missing'
    # Versions can change solver defaults and search behavior without a code change.
    for package, version in versions.items():
        if saved.get('versions', {}).get(package) != version:
            return False, f'{package} version differs or is missing'
    if 'model_settings' not in old:
        return (True, 'legacy metadata explicitly accepted') if accept_legacy else (False, 'legacy model settings missing; use --accept-legacy-results after review')
    if old['model_settings'] != identity['model_settings']:
        return False, 'model settings differ'
    return True, 'compatible'


def find_completed(index, identity, versions, budgets, accept_legacy=False):
    rejected = []
    for path, result in reversed(index.get(combination(identity), [])):
        ok, reason = compatible(result, identity, versions, budgets, accept_legacy)
        if ok:
            return (path, result, reason), rejected
        rejected.append((path.name, reason))
    return None, rejected
