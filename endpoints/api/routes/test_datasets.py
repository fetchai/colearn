from fastapi.testclient import TestClient
import unittest

from api.database import DBDataset, DBModel, DBExperiment
from api.main import app


class APITest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        for model in [DBDataset, DBModel, DBExperiment]:
            print(model.delete().execute())
        assert len(DBDataset.select()[:]) == 0

    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {'state': 'alive and kicking!'}

    def test_create_dataset(self):
        test_dataset = dict(name="foo",
                            loader=dict(name="bar1", params=dict(key="value")
                                        ),
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

        response = self.client.get('/datasets/foo/')
        assert response.status_code == 200
        assert response.json() == test_dataset

        # todo: list datasets

        response = self.client.delete('/datasets/foo/')
        assert response.status_code == 200
        assert response.json() == {}
