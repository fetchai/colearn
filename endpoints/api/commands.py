from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Tuple


@dataclass
class EventRequest:
    event: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class EventResponse:
    success: bool
    error_message: Optional[str] = None


_event_callback: Optional[Callable[[EventRequest], EventResponse]] = None


def clear_event_handler():
    """
    Clear the event handler callback

    :return: None
    """
    global _event_callback
    _event_callback = None


def set_event_handler(handler: Callable[[EventRequest], EventResponse]):
    """
    Set the event handler callback

    :param handler: The callable object to receive API events
    :return: None
    """
    global _event_callback
    _event_callback = handler


def trigger_event_handler(event: str, *args, **kwargs) -> EventResponse:
    """
    Trigger an event

    :param event: The type of the event to trigger
    :param args: The args associated with the event
    :param kwargs: the kwargs associated with the event
    :return: The event response
    """
    global _event_callback
    if _event_callback is None:
        return EventResponse(success=False, error_message='No callback registered')

    return _event_callback(EventRequest(event=event, args=args, kwargs=kwargs))
