import json
from pathlib import Path
import pandas as pd


def load_results(directory='results/runs'):
    rows = []
    for path in Path(directory).glob('*.json'):
        result = json.loads(path.read_text())
        if result['status'] == 'complete':
            for summary in result['summaries']:
                rows.append({**result['config'], **summary, 'run_id': result['run_id']})
    return pd.DataFrame(rows)
