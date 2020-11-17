from fastapi.testclient import TestClient
import unittest

from api.database import DBDataset, DBModel, DBExperiment
from api.main import app


class APITest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        for model in [DBDataset, DBModel, DBExperiment]:
            print(model.delete().execute())
        assert len(DBModel.select()[:]) == 0

    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {'state': 'alive and kicking!'}

    def test_model(self):
        tmodel = dict(name="foo",
                      model="bar",
                      parameters=dict(k1="v1")
                      )

        response = self.client.post('/models/',
                                    json=tmodel
                                    )
        assert response.status_code == 200
        assert response.json() == {}

        response = self.client.get('/models/foo/')
        assert response.status_code == 200
        assert response.json() == tmodel

        # todo: list

        response = self.client.delete('/models/foo/')
        assert response.status_code == 200
        assert response.json() == {}

    def test_trained_model(self):
        # test trained model creation
        tmodel2 = dict(name="foo2",
                       model="bar",
                       parameters=dict(k1="v1"),
                       weights=dict(k2="v2")
                       )

        response = self.client.post('/models/',
                                    json=tmodel2
                                    )
        assert response.status_code == 200
        assert response.json() == {}

        response = self.client.get('/models/foo2/')
        assert response.status_code == 200
        assert response.json() == tmodel2

        # todo: list datasets

        response = self.client.delete('/models/foo2/')
        assert response.status_code == 200
        assert response.json() == {}
