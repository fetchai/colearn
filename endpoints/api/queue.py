from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class QueueElement:
    experiment: str
    active: bool
    model: str
    dataset: str


# this is the placeholder list that will act
_queue: List[QueueElement] = []
_index: Dict[str, int] = {}


def queue_head() -> Optional[QueueElement]:
    global _queue

    if len(_queue) > 0:
        return _queue[0]
    else:
        return None


def queue_update(experiment: str, active: bool, model: str, dataset: str) -> bool:
    global _queue, _index

    if experiment not in _index:  # pragma: no cover
        return False

    # update the queue element
    _queue[_index[experiment]].active = active
    _queue[_index[experiment]].model = model
    _queue[_index[experiment]].dataset = dataset

    return True


def queue_get(active: Optional[bool] = None, model: Optional[str] = None, dataset: Optional[str] = None):
    global _queue

    def select_from_args(element: QueueElement):
        if active is not None and element.active != active:
            return False
        if model is not None and element.model != model:
            return False
        if dataset is not None and element.dataset != dataset:
            return False

        return True

    return list(filter(select_from_args, _queue))


def queue_push(experiment: str, active: bool, model: str, dataset: str):
    global _queue, _index

    def sort_priority(e: QueueElement):
        return 0 if e.active else 1,

    # add and sort the queue
    _queue.append(QueueElement(experiment=experiment, active=active, model=model, dataset=dataset))
    _queue = sorted(_queue, key=sort_priority)

    # rebuild the index - this is a bit of a trade off, and will be quite expensive if the queue length gets large
    _index = {e.experiment: i for i, e in enumerate(_queue)}


def queue_clear():
    global _queue, _index
    _queue = []
    _index = {}
