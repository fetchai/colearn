import unittest

from fastapi.testclient import TestClient

from api.main import app
from api.queue import queue_clear, queue_push
from api.settings import node_info


class NodeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_node_info(self):
        resp = self.client.get('/node/info/')

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), node_info.dict())


class NodeQueueEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        queue_clear()

    def assertIsPage(self, resp, page: int, total_pages: int):
        self.assertEqual(resp['current_page'], page)
        self.assertEqual(resp['total_pages'], total_pages)
        self.assertEqual(resp['is_start'], page == 0)
        self.assertEqual(resp['is_last'], (page + 1) == total_pages)

    def test_empry_active_queue(self):
        resp = self.client.get('/node/queue/active/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 0)
        self.assertIsPage(resp, 0, 1)

    def test_empty_pending_queue(self):
        resp = self.client.get('/node/queue/pending/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 0)
        self.assertIsPage(resp, 0, 1)

    def test_active_queue(self):
        queue_push('exp-01', True, 'model-1', 'dataset-1')
        queue_push('exp-02', True, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/active/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 2)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertEqual(resp['items'][1], 'exp-02')
        self.assertIsPage(resp, 0, 1)

    def test_pending_queue(self):
        queue_push('exp-01', False, 'model-1', 'dataset-1')
        queue_push('exp-02', False, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/pending/')
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 2)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertEqual(resp['items'][1], 'exp-02')
        self.assertIsPage(resp, 0, 1)

    def test_active_queue_pagination(self):
        queue_push('exp-01', True, 'model-1', 'dataset-1')
        queue_push('exp-02', True, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/active/', params={'page': 0, 'page_size': 1})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertIsPage(resp, 0, 2)

        resp = self.client.get('/node/queue/active/', params={'page': 1, 'page_size': 1})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-02')
        self.assertIsPage(resp, 1, 2)

    def test_pending_queue_pagination(self):
        queue_push('exp-01', False, 'model-1', 'dataset-1')
        queue_push('exp-02', False, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/pending/', params={'page': 0, 'page_size': 1})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertIsPage(resp, 0, 2)

        resp = self.client.get('/node/queue/pending/', params={'page': 1, 'page_size': 1})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-02')
        self.assertIsPage(resp, 1, 2)

    def test_active_queue_filtering_by_model(self):
        queue_push('exp-01', True, 'model-1', 'dataset-1')
        queue_push('exp-02', True, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/active/', params={'model': 'model-1'})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertIsPage(resp, 0, 1)

    def test_pending_queue_filtering_by_model(self):
        queue_push('exp-01', False, 'model-1', 'dataset-1')
        queue_push('exp-02', False, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/pending/', params={'model': 'model-2'})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-02')
        self.assertIsPage(resp, 0, 1)

    def test_active_queue_filtering_by_dataset(self):
        queue_push('exp-01', True, 'model-1', 'dataset-1')
        queue_push('exp-02', True, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/active/', params={'dataset': 'dataset-1'})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-01')
        self.assertIsPage(resp, 0, 1)

    def test_pending_queue_filtering_by_dataset(self):
        queue_push('exp-01', False, 'model-1', 'dataset-1')
        queue_push('exp-02', False, 'model-2', 'dataset-2')

        resp = self.client.get('/node/queue/pending/', params={'dataset': 'dataset-2'})
        self.assertEqual(resp.status_code, 200)
        resp = resp.json()

        self.assertEqual(len(resp['items']), 1)
        self.assertEqual(resp['items'][0], 'exp-02')
        self.assertIsPage(resp, 0, 1)
