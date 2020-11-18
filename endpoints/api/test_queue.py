import unittest
from typing import Optional

from .queue import queue_clear, queue_get, queue_push, queue_head, queue_update, QueueElement
from .utils import default


class QueueTests(unittest.TestCase):
    def setUp(self):
        queue_clear()

    @staticmethod
    def addNonActiveElement():
        queue_push('exp-01', False, 'model-1', 'dataset-1')

    def assertIsElement(self, v: QueueElement, experiment: Optional[str] = None, active: Optional[bool] = None,
                        model: Optional[str] = None, dataset: Optional[str] = None):
        self.assertIsNot(v, None)
        self.assertEqual(v.experiment, default(experiment, 'exp-01'))
        self.assertEqual(v.active, default(active, False))
        self.assertEqual(v.model, default(model, 'model-1'))
        self.assertEqual(v.dataset, default(dataset, 'dataset-1'))

    def test_elements_can_be_added(self):
        self.assertEqual(len(queue_get()), 0)

        self.addNonActiveElement()

        q = queue_get()
        self.assertEqual(len(q), 1)
        self.assertIsElement(q[0])

    def test_queue_has_head(self):
        head = queue_head()
        self.assertIs(head, None)

        self.addNonActiveElement()

        head = queue_head()
        self.assertIsNot(head, None)
        self.assertIsElement(head)

    def test_queue_can_be_updated(self):
        self.addNonActiveElement()
        self.assertIsElement(queue_head())

        self.assertTrue(queue_update('exp-01', True, 'model-2', 'dataset-2'))

        self.assertIsElement(queue_head(), active=True, model='model-2', dataset='dataset-2')

    def test_queue_adds_are_sorted(self):
        queue_push(experiment='A', active=False, model='A', dataset='A')
        queue_push(experiment='B', active=True, model='B', dataset='B')

        q = queue_get()
        self.assertEqual(len(q), 2)
        self.assertEqual(q[0].experiment, 'B')
        self.assertEqual(q[1].experiment, 'A')

    def test_queue_filtering_by_active(self):
        queue_push(experiment='A', active=False, model='A', dataset='A')
        queue_push(experiment='B', active=True, model='B', dataset='B')

        q = queue_get(active=True)
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'B')
        self.assertEqual(q[0].active, True)

        q = queue_get(active=False)
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'A')
        self.assertEqual(q[0].active, False)

    def test_queue_filtering_by_model(self):
        queue_push(experiment='A', active=False, model='A', dataset='A')
        queue_push(experiment='B', active=True, model='B', dataset='B')

        q = queue_get(model='A')
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'A')
        self.assertEqual(q[0].model, 'A')

        q = queue_get(model='B')
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'B')
        self.assertEqual(q[0].model, 'B')

        q = queue_get(model='C')
        self.assertEqual(len(q), 0)

    def test_queue_filtering_by_dataset(self):
        queue_push(experiment='A', active=False, model='A', dataset='A')
        queue_push(experiment='B', active=True, model='B', dataset='B')

        q = queue_get(dataset='A')
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'A')
        self.assertEqual(q[0].dataset, 'A')

        q = queue_get(dataset='B')
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].experiment, 'B')
        self.assertEqual(q[0].dataset, 'B')

        q = queue_get(dataset='C')
        self.assertEqual(len(q), 0)
