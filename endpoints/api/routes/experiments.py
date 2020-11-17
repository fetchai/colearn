from typing import Optional

from fastapi import APIRouter, Path

from api.schemas import ExperimentList, Experiment, Status, PerformanceList, VoteList, Statistics, Empty

router = APIRouter()


@router.get('/experiments/', response_model=ExperimentList, tags=['experiments'])
def get_the_list_of_experiments(model: Optional[str] = None, dataset: Optional[str] = None, page: Optional[int] = None,
                                page_size: Optional[int] = None):
    """
    Get a list of the experiments that are present on the system

    Optional parameters:

    * `model` - Filter the results by experiments which use the specified model
    * `dataset` - Filter the results by experiments which use the specific dataset
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return {}


@router.get('/experiments/{name}/', response_model=Experiment, tags=['experiments'])
def get_specific_experiment(name: str):
    """
    Lookup information about a specific experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@router.post('/experiments/{name}/', response_model=Experiment, tags=['experiments'])
def update_specific_experiment(name: str, experiment: Experiment):
    """
    Update the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """
    return {}


@router.delete('/experiments/{name}/', tags=['experiments'])
def delete_specific_experiment(name: str):
    """
    Delete the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """
    return {}


@router.post('/experiments/', response_model=Experiment, tags=['experiments'])
def create_a_new_experiment():
    """
    Create a new experiment
    """
    return {}


@router.get('/experiments/{name}/status/', response_model=Status, tags=['experiments'])
def get_learner_status(name: str):
    """
    Get the status of the current experiment (if there is one). This information is expected to be updated frequently
    as the experiment proceeds.

    It is expected that clients will poll this API endpoint to collect up to date status information for the experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@router.get('/experiments/{name}/performance/{mode}/', response_model=PerformanceList, tags=['experiments'])
def get_performance(name: str, mode: str = Path(..., regex=r'(?:validation|test)'), start: Optional[int] = None,
                    end: Optional[int] = None, page: Optional[int] = None,
                    page_size: Optional[int] = None):
    """
    Queries the performance for a specific mode. This query can provide both current and historical information on the
    currently evaluated model

    The mode is expected to be one of the following:

    * `validation`
    * `test`

    Route parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)

    Optional parameters:

    * `start` - the starting epoch number to return values from
    * `end` - the last epoch number to return values from
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return {}


@router.get('/experiments/{name}/votes/', response_model=VoteList, tags=['experiments'])
def get_vote_information(name: str, start: Optional[int] = None, end: Optional[int] = None, page: Optional[int] = None,
                         page_size: Optional[int] = None):
    """
    Queries the vote information, both present and historical.

    Route parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)

    Optional parameters:

    * `start` - the starting epoch number to return values from
    * `end` - the last epoch number to return values from
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.

    """
    return {}


@router.get('/experiments/{name}/stats/', response_model=Statistics, tags=['experiments'])
def get_learner_statistics(name: str):
    """
    Query basic statistics about the current experiment

    Route parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@router.post('/experiments/{name}/start/', response_model=Empty, tags=['experiments'])
def start_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    return {}


@router.post('/experiments/{name}/join/', response_model=Empty, tags=['experiments'])
def join_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    return {}


@router.post('/experiments/{name}/leave/', response_model=Empty, tags=['experiments'])
def leave_the_current_experiment(name: str):
    """
    Leave the current experiment

    Route Parameters:

    * `name` - The name of the experiment to leave
    """
    return {}


@router.post('/experiments/{name}/stop/', response_model=Empty, tags=['experiments'])
def stop_the_current_experiment(name: str):
    """
    Stop the current experiment. This also prohibts other users from participating in the experiment

    Route Parameters:

    * `name` - The name of the experiment to stop
    """
    return {}
