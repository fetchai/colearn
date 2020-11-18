import unittest
from typing import Optional

from fastapi.testclient import TestClient

from api.commands import EventRequest, EventResponse, set_event_handler
from api.main import app


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # event handling
        self.expected_event: Optional[str] = None
        self.response: Optional[EventResponse] = None
        set_event_handler(self.event_handler)

    def event_handler(self, request: EventRequest) -> EventResponse:
        self.assertIsNotNone(self.expected_event)
        self.assertIsNotNone(self.response)
        self.assertEqual(request.event, self.expected_event)

        resp = self.response

        # clear the pending state
        self.response = None
        self.expected_event = None

        return resp

    def set_successful_event(self, expected_event: str):
        self.expected_event = expected_event
        self.response = EventResponse(success=True)

    def set_unsuccessful_trigger(self, expected_event: str, message: str):
        self.expected_event = expected_event
        self.response = EventResponse(success=False, error_message=message)

    def test_start_experiment_success(self):
        self.set_successful_event('start-experiment')

        resp = self.client.post('/experiments/foo/start/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    def test_start_experiment_failure(self):
        self.set_unsuccessful_trigger('start-experiment', 'Not supported!')

        resp = self.client.post('/experiments/foo/start/')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {'detail': 'Not supported!'})

    def test_join_experiment_success(self):
        self.set_successful_event('join-experiment')

        resp = self.client.post('/experiments/foo/join/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    def test_join_experiment_failure(self):
        self.set_unsuccessful_trigger('join-experiment', 'Not supported!')

        resp = self.client.post('/experiments/foo/join/')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {'detail': 'Not supported!'})

    def test_leave_experiment_success(self):
        self.set_successful_event('leave-experiment')

        resp = self.client.post('/experiments/foo/leave/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    def test_leave_experiment_failure(self):
        self.set_unsuccessful_trigger('leave-experiment', 'Not supported!')

        resp = self.client.post('/experiments/foo/leave/')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {'detail': 'Not supported!'})

    def test_stop_experiment_success(self):
        self.set_successful_event('stop-experiment')

        resp = self.client.post('/experiments/foo/stop/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    def test_stop_experiment_failure(self):
        self.set_unsuccessful_trigger('stop-experiment', 'Not supported!')

        resp = self.client.post('/experiments/foo/stop/')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {'detail': 'Not supported!'})