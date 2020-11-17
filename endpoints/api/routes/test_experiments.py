import json
from typing import Any, Optional

import peewee
from fastapi.testclient import TestClient

from api.database import DBDataset, DBModel, DBExperiment
from api.main import app
from api.schemas import ExperimentParameters, Statistics, Statistic
from api.utils import BasicEndpointTest, default


class ExperimentEndpointTests(BasicEndpointTest):
    def setUp(self):
        self.client = TestClient(app)

        # clear the existing databases
        for model in [DBDataset, DBModel, DBExperiment]:
            model.delete().execute()

    def assertIsExperiment(self, experiment: Any, ref: DBExperiment):
        self.assertIsNot(experiment, None)
        self.assertEqual(experiment['name'], ref.name)

    @staticmethod
    def create_sample_experiment(index: Optional[int] = None, is_owner: Optional[bool] = None):
        index = default(index, 0)
        is_owner = default(is_owner, False)

        model = DBModel.create(
            name=f'model-{index}',
            model='mnist',
            parameters='{}',
            weights=None,
        )

        dataset = DBDataset.create(
            name=f'dataset-{index}',
            loader_name='mnist',
            loader_params='{}',
            location='',
            seed=None,
            train_size=0.5,
            validation_size=None,
            test_size=0.5,
        )

        experiment = DBExperiment.create(
            name=f'exp-{index:02}',
            model=model,
            dataset=dataset,
            seed=None,
            contract_address=None if is_owner else '0xdeadbeefdeadbeefdeadbeef',
            parameters=json.dumps({
                'min_learners': 5,
                'max_learners': 5,
                'num_epochs': 50,
            }),
            is_owner=is_owner,
        )

        return experiment

    def test_list_experiments(self):
        experiment = self.create_sample_experiment()

        resp = self.client.get('/experiments/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertIsExperiment(resp['items'][0], experiment)
        self.assertIsPage(resp, 0, 1)

    def test_empty_list_experiments(self):
        resp = self.client.get('/experiments/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 0)
        self.assertIsPage(resp, 0, 1)

    def test_list_of_experiments_filtered_by_model(self):
        experiment1 = self.create_sample_experiment(1)
        _experiment2 = self.create_sample_experiment(2)

        resp = self.client.get('/experiments/', params={'model': experiment1.model.name})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertIsExperiment(resp['items'][0], experiment1)
        self.assertIsPage(resp, 0, 1)

    def test_list_of_experiments_filtered_by_dataset(self):
        experiment1 = self.create_sample_experiment(1)
        _experiment2 = self.create_sample_experiment(2)

        resp = self.client.get('/experiments/', params={'dataset': experiment1.dataset.name})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertIsExperiment(resp['items'][0], experiment1)
        self.assertIsPage(resp, 0, 1)

    def test_list_of_experiments_filtered_by_dataset_and_model(self):
        experiment1 = self.create_sample_experiment(1)
        _experiment2 = self.create_sample_experiment(2)

        resp = self.client.get('/experiments/', params={
            'dataset': experiment1.dataset.name,
            'model': experiment1.model.name,
        })
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertIsExperiment(resp['items'][0], experiment1)
        self.assertIsPage(resp, 0, 1)

    def test_lookup_failure_for_single_experiment(self):
        resp = self.client.get('/experiments/foo/')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment not found'})

    def test_experiment_delete(self):
        experiment = self.create_sample_experiment()

        resp = self.client.delete(f'/experiments/{experiment.name}/')
        self.assertEqual(resp.status_code, 200)

        with self.assertRaises(peewee.DoesNotExist):
            DBExperiment.get(DBExperiment.name == experiment.name)

    def test_experiment_delete_not_present(self):
        resp = self.client.delete('/experiments/foo/')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment not found'})

    def test_experiment_owner_creation(self):
        experiment = self.create_sample_experiment()

        resp = self.client.post('/experiments/', json={
            'mode': 'owner',
            'name': 'foo',
            'model': experiment.model.name,
            'dataset': experiment.dataset.name,
            'parameters': {
                'min_learners': 5,
                'max_learners': 5,
                'num_epochs': 50,
            }
        })
        self.assertEqual(resp.status_code, 201)
        self.assertIsExperiment(resp.json(), DBExperiment.get(DBExperiment.name == 'foo'))

    def test_experiment_follower_creation(self):
        experiment = self.create_sample_experiment()

        resp = self.client.post('/experiments/', json={
            'mode': 'follower',
            'name': 'foo',
            'model': experiment.model.name,
            'dataset': experiment.dataset.name,
            'contract_address': '0xasdasdasdsadsad',
            'parameters': {
                'min_learners': 5,
                'max_learners': 5,
                'num_epochs': 50,
            }
        })
        self.assertEqual(resp.status_code, 201)
        self.assertIsExperiment(resp.json(), DBExperiment.get(DBExperiment.name == 'foo'))

    def test_experiment_owner_creation_failure(self):
        resp = self.client.post('/experiments/', json={
            'mode': 'owner',
            'name': 'foo',
            'model': 'does-not-exist',
            'dataset': 'does-not-exist',
            'parameters': {
                'min_learners': 5,
                'max_learners': 5,
                'num_epochs': 50,
            }
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment and/or Model and/or Dataset not found'})

    def test_experiment_follower_creation_failure(self):
        resp = self.client.post('/experiments/', json={
            'mode': 'follower',
            'name': 'foo',
            'model': 'does-not-exist',
            'dataset': 'does-not-exist',
            'contract_address': '0xasdasdasdsadsad',
            'parameters': {
                'min_learners': 5,
                'max_learners': 5,
                'num_epochs': 50,
            }
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment and/or Model and/or Dataset not found'})

    def test_owner_experiment_update(self):
        experiment = self.create_sample_experiment(is_owner=True)

        resp = self.client.post(f'/experiments/{experiment.name}/', json={
            'training_mode': 'foobar',
            'model': experiment.model.name,
            'dataset': experiment.dataset.name,
            'seed': 42,
            'parameters': {
                'min_learners': 6,
                'max_learners': 6,
                'num_epochs': 100,
            }
        })
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        # lookup the updated record
        record: DBExperiment = DBExperiment.get(DBExperiment.name == experiment.name)
        self.assertIsExperiment(resp, record)  # check they match

        # check the values were actually updated
        self.assertEqual(record.training_mode, 'foobar')
        self.assertEqual(record.model.name, experiment.model.name)
        self.assertEqual(record.dataset.name, experiment.dataset.name)
        self.assertEqual(record.seed, 42)
        self.assertEqual(json.loads(record.parameters), ExperimentParameters(
            min_learners=6,
            max_learners=6,
            num_epochs=100,
        ).dict())

    def test_follower_experiment_update(self):
        experiment = self.create_sample_experiment()

        resp = self.client.post(f'/experiments/{experiment.name}/', json={
            'training_mode': 'foobar',
            'model': experiment.model.name,
            'dataset': experiment.dataset.name,
            'contract_address': '0x12341234123412341234',
            'seed': 42,
            'parameters': {
                'min_learners': 6,
                'max_learners': 6,
                'num_epochs': 100,
            }
        })
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        # lookup the updated record
        record: DBExperiment = DBExperiment.get(DBExperiment.name == experiment.name)
        self.assertIsExperiment(resp, record)  # check they match

        # check the values were actually updated
        self.assertEqual(record.training_mode, 'foobar')
        self.assertEqual(record.model.name, experiment.model.name)
        self.assertEqual(record.dataset.name, experiment.dataset.name)
        self.assertEqual(record.contract_address, '0x12341234123412341234')
        self.assertEqual(record.seed, 42)
        self.assertEqual(json.loads(record.parameters), ExperimentParameters(
            min_learners=6,
            max_learners=6,
            num_epochs=100,
        ).dict())

    def test_invalid_owner_experiment_update(self):
        experiment = self.create_sample_experiment(is_owner=True)

        # owner can't directly update the contract address (it will be provided by the system)
        resp = self.client.post(f'/experiments/{experiment.name}/', json={
            'contract_address': '0x12341234123412341234',
        })
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {'detail': 'Unable to update joining contract address'})

    def test_invalid_experiment_update(self):
        # try and update an experiment that does not exist
        resp = self.client.post(f'/experiments/foo/', json={
            'contract_address': '0x12341234123412341234',
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment and/or Model and/or Dataset not found'})

    def test_get_experiment_status(self):
        experiment = self.create_sample_experiment()
        experiment.epoch = 5
        experiment.state = 'training'
        experiment.save()

        resp = self.client.get(f'/experiments/{experiment.name}/status/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {
            'experiment': experiment.name,
            'epoch': 5,
            'state': 'training',
        })

    def test_get_experiment_status_failure(self):
        resp = self.client.get('/experiments/foo/status/')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment not found'})

    def test_get_experiment_stats(self):
        experiment = self.create_sample_experiment()
        experiment.mean_epoch_time = 1.2
        experiment.mean_train_time = 1.3
        experiment.mean_evaluation_time = 1.4
        experiment.save()

        resp = self.client.get(f'/experiments/{experiment.name}/stats/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), Statistics(
            epoch_time=Statistic(mean=1.2),
            train_time=Statistic(mean=1.3),
            evaluate_time=Statistic(mean=1.4),
        ).dict())

    def test_get_experiment_stats_not_present(self):
        experiment = self.create_sample_experiment()

        resp = self.client.get(f'/experiments/{experiment.name}/stats/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), Statistics(
            epoch_time=Statistic(mean=0.0),
            train_time=Statistic(mean=0.0),
            evaluate_time=Statistic(mean=0.0),
        ).dict())

    def test_get_experiment_stats_failure(self):
        resp = self.client.get('/experiments/foo/stats/')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Experiment not found'})
