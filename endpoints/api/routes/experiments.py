import json
from typing import Optional

import peewee
from fastapi import APIRouter, Path, HTTPException

from api.commands import trigger_event_handler
from api.database import DBExperiment, DBModel, DBDataset, DBPerformance, DBVote
from api.schemas import ExperimentList, Experiment, Status, PerformanceList, VoteList, Statistics, Empty, \
    ExperimentParameters, ErrorResponse, CreateExperiment, UpdateExperiment, Statistic, Performance, Vote
from api.utils import paginate_db, aggregate_conditions

router = APIRouter()


def _convert_vote(record: DBVote) -> Vote:
    return Vote(
        epoch=record.epoch,
        vote=record.vote,
        is_proposer=record.is_proposer,
    )


def _convert_perf(record: DBPerformance) -> Performance:
    return Performance(
        epoch=record.epoch,
        performance=record.performance,
    )


def _convert(record: DBExperiment) -> Experiment:
    return Experiment(
        name=record.name,
        model=record.model.name,
        dataset=record.dataset.name,
        seed=record.seed,
        contract_address=record.contract_address,
        parameters=ExperimentParameters(
            **json.loads(record.parameters)
        ),
        is_owner=record.is_owner,
    )


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
      specified, however, it might respond with fewer.
    """

    # build up the database query
    query = DBExperiment.select()

    conditions = []
    if model is not None:
        conditions.append(DBExperiment.model == model)
    if dataset is not None:
        conditions.append(DBExperiment.dataset == dataset)

    if len(conditions) > 0:
        query = query.where(aggregate_conditions(conditions))

    return paginate_db(query, ExperimentList, _convert, page, page_size)


@router.get(
    '/experiments/{name}/',
    response_model=Experiment,
    tags=['experiments'],
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
    }
)
def get_specific_experiment(name: str):
    """
    Lookup information about a specific experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """

    try:
        return _convert(DBExperiment.get(DBExperiment.name == name))
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment not found")


@router.post(
    '/experiments/{name}/',
    response_model=Experiment,
    tags=['experiments'],
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
        409: {"description": "Experiment state conflict", 'model': ErrorResponse},
    }
)
def update_specific_experiment(name: str, experiment: UpdateExperiment):
    """
    Update the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """

    try:
        exp: DBExperiment = DBExperiment.get(DBExperiment.name == name)

        if experiment.training_mode is not None:
            exp.training_mode = experiment.training_mode
        if experiment.model is not None:
            exp.model = DBModel.get(DBModel.name == experiment.model)
        if experiment.dataset is not None:
            exp.dataset = DBDataset.get(DBDataset.name == experiment.dataset)
        if experiment.seed is not None:
            exp.seed = experiment.seed
        if experiment.contract_address is not None:
            if exp.is_owner:
                raise HTTPException(status_code=409, detail="Unable to update joining contract address")
            exp.contract_address = experiment.contract_address
        if experiment.parameters is not None:
            exp.parameters = experiment.parameters.json()

        exp.save()

        return _convert(exp)

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment and/or Model and/or Dataset not found")


@router.delete(
    '/experiments/{name}/',
    response_model=Empty,
    tags=['experiments'],
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
    }
)
def delete_specific_experiment(name: str):
    """
    Delete the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """
    try:
        experiment = DBExperiment.get(DBExperiment.name == name)
        experiment.delete_instance()
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {}


@router.post(
    '/experiments/',
    tags=['experiments'],
    status_code=201,
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
    }
)
def create_a_new_experiment(experiment: CreateExperiment):
    """
    Create a new experiment
    """
    try:
        params = dict(
            name=experiment.name,
            training_mode=experiment.training_mode,
            model=DBModel.get(DBModel.name == experiment.model),
            dataset=DBDataset.get(DBDataset.name == experiment.dataset),
            seed=experiment.seed,
        )

        if experiment.mode == 'owner':
            params.update(dict(
                contract_address=None,
                parameters=experiment.parameters.json(),
                is_owner=True,
            ))
        elif experiment.mode == 'follower':
            params.update(dict(
                contract_address=experiment.contract_address,
                parameters=experiment.parameters.json(),
                is_owner=False,
            ))
        else:
            raise HTTPException(status_code=500, detail="Logical error")  # pragma: no cover

        # create the new database entry
        record = DBExperiment.create(**params)

        return _convert(record)

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment and/or Model and/or Dataset not found")


@router.get(
    '/experiments/{name}/status/',
    response_model=Status,
    tags=['experiments'],
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
    }
)
def get_learner_status(name: str):
    """
    Get the status of the current experiment (if there is one). This information is expected to be updated frequently
    as the experiment proceeds.

    It is expected that clients will poll this API endpoint to collect up to date status information for the experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    try:
        exp: DBExperiment = DBExperiment.get(DBExperiment.name == name)

        return Status(
            experiment=exp.name,
            state=exp.state,
            epoch=exp.epoch,
        )

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment not found")


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
      specified, however, it might respond with fewer.
    """

    # build up the conditions
    conditions = [DBPerformance.experiment == name, DBPerformance.mode == mode]
    if start is not None:
        conditions.append(DBPerformance.epoch >= start)
    if end is not None:
        conditions.append(DBPerformance.epoch <= end)

    query = DBPerformance.select().where(aggregate_conditions(conditions))

    return paginate_db(query, PerformanceList, _convert_perf, page, page_size)


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
      specified, however, it might respond with fewer.

    """

    # build up the conditions
    conditions = [DBVote.experiment == name]
    if start is not None:
        conditions.append(DBVote.epoch >= start)
    if end is not None:
        conditions.append(DBVote.epoch <= end)

    # build the final query
    query = DBVote.select().where(aggregate_conditions(conditions)).order_by(DBVote.epoch)

    return paginate_db(query, VoteList, _convert_vote, page, page_size)


@router.get(
    '/experiments/{name}/stats/',
    response_model=Statistics,
    tags=['experiments'],
    responses={
        404: {"description": "Experiment not found", 'model': ErrorResponse},
    }
)
def get_learner_statistics(name: str):
    """
    Query basic statistics about the current experiment

    Route parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    try:
        exp: DBExperiment = DBExperiment.get(DBExperiment.name == name)

        return Statistics(
            epoch_time=Statistic(mean=exp.mean_epoch_time or 0.0),
            train_time=Statistic(mean=exp.mean_train_time or 0.0),
            evaluate_time=Statistic(mean=exp.mean_evaluation_time or 0.0),
        )

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment not found")


@router.post(
    '/experiments/{name}/start/',
    response_model=Empty,
    tags=['experiments'],
    responses={
        409: {"description": "Operation not valid", 'model': ErrorResponse},
    }
)
def start_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    response = trigger_event_handler('start-experiment')
    if not response.success:
        raise HTTPException(status_code=409, detail=response.error_message)

    return {}


@router.post(
    '/experiments/{name}/join/',
    response_model=Empty,
    tags=['experiments'],
    responses={
        409: {"description": "Operation not valid", 'model': ErrorResponse},
    }
)
def join_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    response = trigger_event_handler('join-experiment')
    if not response.success:
        raise HTTPException(status_code=409, detail=response.error_message)

    return {}


@router.post(
    '/experiments/{name}/leave/',
    response_model=Empty,
    tags=['experiments'],
    responses={
        409: {"description": "Operation not valid", 'model': ErrorResponse},
    }
)
def leave_the_current_experiment(name: str):
    """
    Leave the current experiment

    Route Parameters:

    * `name` - The name of the experiment to leave
    """
    response = trigger_event_handler('leave-experiment')
    if not response.success:
        raise HTTPException(status_code=409, detail=response.error_message)

    return {}


@router.post(
    '/experiments/{name}/stop/',
    response_model=Empty,
    tags=['experiments'],
    responses={
        409: {"description": "Operation not valid", 'model': ErrorResponse},
    }
)
def stop_the_current_experiment(name: str):
    """
    Stop the current experiment. This also prohibts other users from participating in the experiment

    Route Parameters:

    * `name` - The name of the experiment to stop
    """
    response = trigger_event_handler('stop-experiment')
    if not response.success:
        raise HTTPException(status_code=409, detail=response.error_message)

    return {}
