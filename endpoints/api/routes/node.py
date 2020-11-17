from typing import Optional

from fastapi import APIRouter

from api.queue import queue_get
from api.schemas import Info, QueueList
from api.settings import node_info
from api.utils import paginate

router = APIRouter()


@router.get('/node/info/', response_model=Info, tags=['node'])
def get_learner_information():
    """
    Get the static learner information. This is information that is not expected to change for the lifetime of the
    learner.
    """
    return node_info


@router.get('/node/queue/active/', response_model=QueueList, tags=['node'])
def get_learner_active_queue(model: Optional[str] = None, dataset: Optional[str] = None, page: Optional[int] = None,
                             page_size: Optional[int] = None):
    """
    Lookup the experiments that are currently being trained by the learner

    Optional parameters:

    * `model` - Filter the results by the specified model
    * `dataset` - Filter the results by the specified dataset
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return paginate(
        QueueList,
        map(
            lambda x: x.experiment,
            queue_get(dataset=dataset, model=model, active=True)
        ),
        page,
        page_size
    )


@router.get('/node/queue/pending/', response_model=QueueList, tags=['node'])
def get_learner_pending_queue(model: Optional[str] = None, dataset: Optional[str] = None, page: Optional[int] = None,
                              page_size: Optional[int] = None):
    """
    Lookup the experiments that are currently waiting to be trained by the learner

    Optional parameters:

    * `model` - Filter the results by the specified model
    * `dataset` - Filter the results by the specified dataset
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return paginate(
        QueueList,
        map(
            lambda x: x.experiment,
            queue_get(dataset=dataset, model=model, active=False)
        ),
        page,
        page_size
    )
