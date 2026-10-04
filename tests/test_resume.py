import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from hpo_project.models import make_model
from hpo_project.optimizers import optimize
from hpo_project.resume import compatible, index_results, find_completed, model_settings
from hpo_project.runner import run


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(dataset='synthetic_linear',condition='base',model='elastic_net',optimizer='random',budget=100,seed=1,
            dataset_spec={'task':'regression'},space={'alpha':{'low':.01,'high':1}},cv_folds=5,protocol_version=1,
            model_settings=model_settings(make_model('elastic_net','regression',1,1)))
        self.versions={'scikit-learn':'test'}
        self.saved=dict(status='complete',config=copy.deepcopy(self.identity),versions=self.versions,
                        trials=[{'score':1}],summaries=[{'budget':25},{'budget':100}])

    def test_code_and_parallelism_changes_do_not_invalidate_scores(self):
        old=self.saved['config'];old.update(source_hash='old',code_revision='old',n_jobs=1)
        new={**self.identity,'source_hash':'new','code_revision':'new','cv_jobs':4}
        self.assertTrue(compatible(self.saved,new,self.versions,[25,100])[0])

    def test_protocol_and_model_changes_rejected(self):
        for key in ['dataset_spec','space','cv_folds','protocol_version','model_settings']:
            with self.subTest(key=key):
                identity={**self.identity,key:'changed'}
                self.assertFalse(compatible(self.saved,identity,self.versions,[25,100],True)[0])

    def test_legacy_requires_explicit_acceptance(self):
        del self.saved['config']['model_settings']
        self.assertFalse(compatible(self.saved,self.identity,self.versions,[25,100])[0])
        self.assertTrue(compatible(self.saved,self.identity,self.versions,[25,100],True)[0])

    def test_versions_and_missing_budgets_rejected(self):
        self.assertFalse(compatible(self.saved,self.identity,{'scikit-learn':'new'},[25,100],True)[0])
        self.saved['summaries']=[{'budget':100}]
        self.assertFalse(compatible(self.saved,self.identity,self.versions,[25,100],True)[0])

    def test_index_ignores_incomplete_and_finds_across_filenames(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'old.json').write_text(json.dumps(self.saved))
            interrupted={**self.saved,'status':'interrupted'}
            Path(folder,'new.json').write_text(json.dumps(interrupted))
            found,_=find_completed(index_results(folder),self.identity,self.versions,[25,100])
            self.assertEqual(found[0].name,'old.json')

    def test_dry_run_skip_and_force_without_training(self):
        config=yaml.safe_load(Path('configs/experiments.yaml').read_text())
        config.update(seeds=[1],budgets=[2],optimizers=['random'],cv_folds=2)
        spaces=yaml.safe_load(Path('configs/search_spaces.yaml').read_text())
        selection=dict(dataset='synthetic_linear',condition='base',model='elastic_net',optimizer='random',seed=1)
        with tempfile.TemporaryDirectory() as folder:
            first=run(config,spaces,selection,output=folder)
            files=list(Path(folder).glob('*.json'))
            with patch('hpo_project.runner.load_dataset',side_effect=AssertionError('Must not load data')):
                self.assertEqual(run(config,spaces,selection,output=folder)['skip'],1)
                self.assertEqual(run(config,spaces,selection,output=folder,dry_run=True,force=True)['run'],1)
            second=run(config,spaces,selection,output=folder,force=True)
            self.assertEqual(len(list(Path(folder).glob('*.json'))),2)
            self.assertTrue(files[0].exists())

    def test_parallel_halving_matches_sequential_and_reports_progress(self):
        X,y=make_regression(n_samples=90,n_features=5,random_state=1)
        X,y=pd.DataFrame(X),pd.Series(y)
        folds=list(KFold(3).split(X))
        estimator=make_model('elastic_net','regression',1,1)
        space={'model__alpha':{'low':.01,'high':1}}
        a=optimize(estimator,space,'halving',2,X,y,folds,'neg_root_mean_squared_error',1,1)
        calls=[]
        b=optimize(estimator,space,'halving',2,X,y,folds,'neg_root_mean_squared_error',1,2,progress=lambda ts:calls.append(len(ts)))
        np.testing.assert_allclose([t['score'] for t in a],[t['score'] for t in b])
        self.assertEqual(calls,list(range(1,len(b)+1)))

    def test_interrupt_preserves_trial_checkpoint_and_restarts(self):
        config=yaml.safe_load(Path('configs/experiments.yaml').read_text())
        config.update(seeds=[1],budgets=[2],optimizers=['random'],cv_folds=2)
        spaces=yaml.safe_load(Path('configs/search_spaces.yaml').read_text())
        selection=dict(dataset='synthetic_linear',condition='base',model='elastic_net',optimizer='random',seed=1)
        def interrupt(*args, **kwargs):
            kwargs['progress']([dict(score=-1,resource=1,elapsed_seconds=1)])
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as folder:
            with patch('hpo_project.runner.optimize',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run(config,spaces,selection,output=folder)
            saved=json.loads(next(Path(folder).glob('*.json')).read_text())
            self.assertEqual(saved['status'],'interrupted')
            self.assertEqual(len(saved['trials']),1)
            self.assertEqual(run(config,spaces,selection,output=folder)['run'],1)
