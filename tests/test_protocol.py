import unittest
from pathlib import Path
import yaml
import numpy as np
from hpo_project.datasets import load_dataset, prepare_split
from hpo_project.runner import matrix


class ProtocolTests(unittest.TestCase):
    def test_matrix_counts(self):
        config = yaml.safe_load(Path('configs/experiments.yaml').read_text())
        runs = list(matrix(config))
        self.assertEqual(len(runs), 3060)
        self.assertEqual(len({(r['dataset'], r['condition'], r['model']) for r in runs}), 51)
        self.assertFalse(any(r['model'] == 'svm' and r['dataset'] == 'bank_marketing' and r['condition'] != 'small' for r in runs))

    def test_noise_preserves_clean_test_set_and_split(self):
        X, y = load_dataset('synthetic_balanced', {'task': 'classification'}, 1)
        base = prepare_split(X, y, 'classification', 'base', 1)
        noisy = prepare_split(X, y, 'classification', 'noisy', 1)
        self.assertTrue(base[0].equals(noisy[0]))
        self.assertTrue(base[1].equals(noisy[1]))
        self.assertTrue(base[3].equals(noisy[3]))
        self.assertEqual(np.sum(base[2].to_numpy() != noisy[2].to_numpy()), 160)


if __name__ == '__main__':
    unittest.main()
