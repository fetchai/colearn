import json
from typing import Any, Optional

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

    def assertIsModel(self, model: Any, ref: DBModel):
        self.assertIsNot(model, None)
        self.assertEqual(model['name'], ref.name)
        self.assertEqual(model['model'], ref.model)
        self.assertEqual(model['parameters'], json.loads(ref.parameters))
        try:
            self.assertEqual(model['weights'], json.loads(ref.weights))
        except KeyError:
            pass

    @staticmethod
    def create_sample_model(index: Optional[int] = 0):
        model = DBModel.create(
            name=f'model-{index}',
            model="foo",
            parameters=json.dumps(dict(k1="v1")),
            weights=json.dumps(dict(k2="v2"))
        )
        return model

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

        m1 = DBModel.get(DBModel.name == tmodel['name'])
        self.assertIsModel(tmodel, m1)

    def test_get_model(self):
        model = self.create_sample_model()

        response = self.client.get(f'/models/{model.name}/')
        assert response.status_code == 200
        self.assertIsModel(response.json(), model)

    def test_delete_model(self):
        model = self.create_sample_model()
        response = self.client.delete(f'/models/{model.name}/')
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
