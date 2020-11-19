import json
from typing import Any, Optional

import peewee
from fastapi.testclient import TestClient

from api.database import DBDataset, DBModel, DBExperiment
from api.main import app
from api.utils import BasicEndpointTest


class APITest(BasicEndpointTest):
    def setUp(self):
        self.client = TestClient(app)

        # clear the existing databases
        for model in [DBDataset, DBModel, DBExperiment]:
            model.delete().execute()

    def assertIsDataset(self, dataset: Any, ref: DBDataset):
        self.assertIsNot(dataset, None)
        self.assertEqual(dataset['name'], ref.name)
        try:
            self.assertEqual(dataset['loader_name'], ref.loader_name)
            self.assertEqual(dataset['loader_params'], ref.loader_params)
        except KeyError:
            self.assertEqual(dataset['loader']['name'], ref.loader_name)
            self.assertEqual(dataset['loader']['params'], json.loads(ref.loader_params))

        self.assertEqual(dataset['location'], ref.location)
        self.assertEqual(dataset['seed'], ref.seed)
        self.assertEqual(dataset['train_size'], ref.train_size)
        self.assertEqual(dataset['validation_size'], ref.validation_size)
        self.assertEqual(dataset['test_size'], ref.test_size)

    @staticmethod
    def create_sample_dataset(index: Optional[int] = 0):
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

        return dataset

    def test_create_dataset(self):
        test_dataset = dict(name="foo",
                            loader=dict(name="bar1",
                                        params=dict(key="value")),
                            location="test1",
                            seed=42,
                            train_size=0.6,
                            validation_size=0.3,
                            test_size=0.1
                            )

        response = self.client.post('/datasets/',
                                    json=test_dataset
                                    )
        assert response.status_code == 200
        assert response.json() == {}

        ds1 = DBDataset.get(DBDataset.name == test_dataset['name'])
        self.assertIsDataset(test_dataset, ds1)

    def test_create_dataset_duplicate(self):
        dataset1 = self.create_sample_dataset()

        test_dataset = dict(name=dataset1.name,
                            loader=dict(name="bar1",
                                        params=dict(key="value")),
                            location="test1",
                            seed=42,
                            train_size=0.6,
                            validation_size=0.3,
                            test_size=0.1
                            )

        response = self.client.post('/datasets/',
                                    json=test_dataset
                                    )
        self.assertEqual(response.status_code, 409)
        assert response.json() == {'detail': 'Duplicate dataset name'}

    def test_get_dataset(self):
        dataset = self.create_sample_dataset()

        response = self.client.get(f'/datasets/{dataset.name}/')
        assert response.status_code == 200
        self.assertIsDataset(response.json(), dataset)

    def test_get_dataset_not_present(self):
        resp = self.client.get('/datasets/foo/')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {'detail': 'Dataset not found'})

    def test_delete_dataset(self):
        dataset = self.create_sample_dataset()

        response = self.client.delete(f'/datasets/{dataset.name}/')
        assert response.status_code == 200
        assert response.json() == {}

        with self.assertRaises(peewee.DoesNotExist):
            DBDataset.get(DBDataset.name == dataset.name)

    def test_delete_dataset_not_present(self):
        response = self.client.delete('/datasets/foo/')
        assert response.status_code == 404
        self.assertEqual(response.json(), {'detail': 'Dataset not found'})

    def test_list_dataset(self):
        test_dataset1 = self.create_sample_dataset(1)
        test_dataset2 = self.create_sample_dataset(2)

        response = self.client.get('/datasets/')
        self.assertEqual(response.status_code, 200)
        resp = response.json()
        self.assertEqual(len(resp['items']), 2)
        self.assertIsPage(resp, 0, 1)
        self.assertIsDataset(resp['items'][0], test_dataset1)
        self.assertIsDataset(resp['items'][1], test_dataset2)

    def test_empty_list_dataets(self):
        resp = self.client.get('/datasets/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 0)
        self.assertIsPage(resp, 0, 1)
