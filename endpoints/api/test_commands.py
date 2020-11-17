import unittest
from typing import Optional

from api.commands import EventRequest, EventResponse, trigger_event_handler, set_event_handler


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.expected_request: Optional[EventRequest] = None
        self.response: Optional[EventResponse] = None
        set_event_handler(self.event_handler)

    def event_handler(self, request: EventRequest) -> EventResponse:
        self.assertIsNotNone(self.expected_request)
        self.assertIsNotNone(self.response)
        self.assertEqual(self.expected_request, request)
        return self.response

    def test_triggering_of_event(self):
        self.expected_request = EventRequest(
            event='foo',
            args=(0, 1),
            kwargs={'k': 'v'},
        )
        self.response = EventResponse(
            success=True,
        )

        resp = trigger_event_handler('foo', *self.expected_request.args, **self.expected_request.kwargs)
        self.assertEqual(self.response, resp)
