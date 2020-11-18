from fastapi.testclient import TestClient
import unittest

from api.database import DBDataset, DBModel, DBExperiment
from api.main import app


class APITest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        for model in [DBDataset, DBModel, DBExperiment]:
            model.delete().execute()
        assert len(DBModel.select()[:]) == 0

    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {'state': 'alive and kicking!'}

    def test_creation(self):
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

        response = self.client.delete('/models/foo/')
        assert response.status_code == 200
        assert response.json() == {}

    def test_trained_model_creation(self):
        test_mod = dict(name="foo2",
                        model="bar",
                        parameters=dict(k1="v1"),
                        weights=dict(k2="v2")
                        )

        response = self.client.post('/models/',
                                    json=test_mod
                                    )
        assert response.status_code == 200
        assert response.json() == {}

        response = self.client.get('/models/foo2/')
        assert response.status_code == 200
        original_model = test_mod.copy()
        test_mod.pop("weights")
        assert response.json() == test_mod

        response = self.client.get('/models/foo2/export')
        assert response.status_code == 200
        assert response.json() == original_model

        response = self.client.delete('/models/foo2/')
        assert response.status_code == 200
        assert response.json() == {}

    def test_duplicate_model(self):
        name = "foo"
        test_mod = dict(name=name,
                        model="bar",
                        parameters=dict(k1="v1"),
                        weights=dict(k2="v2")
                        )

        self.client.post('/models/',
                         json=test_mod
                         )

        new_name = "foo10"
        response = self.client.post(f'/models/{name}/copy/',
                                    json=dict(name=new_name, keep_weights=True))
        assert response.status_code == 200

        response = self.client.get(f'/models/{new_name}/export')
        assert response.status_code == 200
        new_mod = response.json()
        assert new_mod["name"] == new_name
        new_mod.pop("name")
        test_mod.pop("name")
        assert new_mod == test_mod

    def test_update_model(self):
        name = "foo"
        test_mod = dict(name=name,
                        model="bar",
                        parameters=dict(k1="v1"),
                        weights=dict(k2="v2")
                        )

        self.client.post('/models/',
                         json=test_mod
                         )

        new_mod = dict(weights=dict(k2="v22"))
        self.client.post(f'/models/{name}/', json=new_mod)

        response = self.client.get(f'/models/{name}/export')
        assert response.status_code == 200
        test_mod.update(new_mod)
        assert response.json() == test_mod

    def test_list_model(self):
        name = "foo"
        test_mod = dict(name=name,
                        model="bar",
                        parameters=dict(k1="v1")
                        )
        self.client.post('/models/',
                         json=test_mod
                         )
        name2 = "foo2"
        test_mod2 = dict(name=name2,
                         model="bar",
                         parameters=dict(k1="v1")
                         )

        self.client.post('/models/',
                         json=test_mod2
                         )

        response = self.client.get('/models/')
        assert response.status_code == 200
        assert response.json() == {'current_page': 0,
                                   'total_pages': 1,
                                   'is_first': True,
                                   'is_last': True,
                                   'items': [test_mod, test_mod2]}
