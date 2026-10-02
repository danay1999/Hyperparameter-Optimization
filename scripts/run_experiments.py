import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from hpo_project.runner import matrix, run

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/experiments.yaml')
parser.add_argument('--spaces', default='configs/search_spaces.yaml')
parser.add_argument('--pilot', action='store_true', help='Synthetic linear / elastic net / seed 1 / budget 4; all four optimizers')
parser.add_argument('--dry-run', action='store_true')
for key in ['dataset', 'condition', 'model', 'optimizer']:
    parser.add_argument(f'--{key}')
parser.add_argument('--seed', type=int)
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
spaces = yaml.safe_load(Path(args.spaces).read_text())
selection = {key: getattr(args, key) for key in ['dataset', 'condition', 'model', 'optimizer', 'seed']}
if args.pilot:
    config.update(seeds=[1], budgets=[4], cv_folds=3, optimizers=['grid', 'random', 'bayesian', 'halving'])
    selection.update(dataset='synthetic_linear', condition='base', model='elastic_net', seed=1)
runs = [item for item in matrix(config) if all(value is None or item[key] == value for key, value in selection.items())]
if args.dry_run:
    print(f'{len(runs)} optimization runs; {len({(x["dataset"], x["condition"], x["model"]) for x in runs})} dataset-condition-model cells')
    for item in runs[:10]:
        print(item)
else:
    run(config, spaces, selection)
