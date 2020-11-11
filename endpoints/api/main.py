from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, Path
from pydantic import BaseModel, validator

app = FastAPI(
    title='Collaborative Learning API',
    description="""
The common collaborative learning API is used for both monitoring and controlling of models, datasets and experiments  
    """,
    version='0.1.0'
)


# ----------------------------------------------------------------------------------------------------------------------
# SCHEMAS
# ----------------------------------------------------------------------------------------------------------------------

class BaseListModel(BaseModel):
    current_page: int
    total_pages: int
    is_start: bool
    is_last: bool


class Loader(BaseModel):
    """
    The loader configuration

    Attributes:

    * `name` - The name of the loader to be used
    * `params` - The loader specific parameters to be used
    """
    name: str
    params: Dict[str, Any]


class Dataset(BaseModel):
    """
    The information to define a dataset that can be trained against

    Attributes:

    * `name` - The name of the dataset
    * `loader` - The path to the dataset
    * `location` - The path to the dataset
    * `seed` - The optional seed value used for dataset splitting
    * `train_size` - The proportion of the dataset to be used for training. (0.0, 1.0)
    * `validation_size` - The optional proportion of the dataset to be used for validation. (0.0, 1.0)
    * `test_size` - The proportion of the dataset to be used for testing. (0.0, 1.0)
    """
    name: str
    loader: Loader
    location: str
    seed: Optional[int]  # range? [1, 100] ? :grin:
    train_size: float
    validation_size: Optional[float]
    test_size: float


class DatasetList(BaseListModel):
    """
    A paged list of datasets

    Attributes:

    * `items` - The list of datasets for this page
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[Dataset]


class Model(BaseModel):
    """
    A model definition

    Attributes:

    * `name` - The name of the model
    * `model` - The type of the model to be used
    * `parameters` - The dictionary of parameters which configure the specified model instance
    """
    name: str
    model: str
    parameters: Dict[str, Any]


class TrainedModel(Model):
    """
    A trained model definition

    Attributes:

    * `name` - The name of the model
    * `model` - The type of the model to be used
    * `parameters` - The dictionary of parameters which configure the model
    * `weights` - A dictionary of weights corresponding to the model
    """
    weights: Dict[str, Any]


class ModelList(BaseListModel):
    """
    A paged list of models

    Attributes:

    * `items` - The list of models for this page
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[Model]


class CopyParams(BaseModel):
    """
    The parameters for the new checkpoint / duplicate model

    Attributes:

    * `name` - The new name for the model
    * `keep_weights` - Flag to signal if the existing weights are kept

    """
    name: str
    keep_weights: bool


class Info(BaseModel):
    """
    Learning information that is expected to not change

    Attributes:

    * `name` - The name of the learner
    * `identity` - The public crypto graphic identity of the learner (if it has one)
    * `driver` - The current driver that is being used by the learner
    * `version` - The current version of the driver being used
    """
    name: str
    identity: str
    driver: str
    version: str


class QueueList(BaseListModel):
    """
    A paged list of queued experiments

    Attributes:

    * `items` - The list of experiment names
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[str]


class ExperimentParameters(BaseModel):
    """
    Experiment Parameters

    Attributes:

    * `explicit_learners` - The list of learner public identities
    * `min_learners` - The minimum number of learners to be part of the experiment epoch
    * `max_learners` - The maximum number of learners that are allowed
    * `num_epochs` - The limit on the number of epochs the training should do
    * `vote_threshold` - The threshold of votes required to accepts weights for an epoch
    """
    explicit_learners: Optional[List[str]]
    min_learners: int
    max_learners: int
    num_epochs: int
    vote_threshold: float = 0.5


class Experiment(BaseModel):
    """
    An experiment definition

    Attributes:

    * `name` - The name of the experiment
    * `training_mode` - The training mode for the experiment. Should be one of the following values:
        - `'collaborative'` (Default)
    * `model` - The name of the model to be used
    * `dataset` - The name of the dataset to be used
    * `seed` - The seed value to initialise the experiment
    * `contract_address` - The contract address for the experiment
    * `parameters` - The experiment parameters
    * `is_owner` - Status field indicating if the current node is the owner of the experiment

    Starting an experiment

    When wishing to start an new experiment the user must populate the `parameters` field and leave empty the
    `contract_address` field. After the experiment has been successfully started the `contract_address` will be
    populated

    Joining an experiment

    When wishing to join an existing experiment the user must populate the `contract_address` field and leave empty the
    `parameters` field. After a successful join of the experiment the parameters field will be automatically populated
    from information downloaded by the contract

    """
    name: str
    training_mode: str = 'collaborative'
    model: str
    dataset: str
    seed: Optional[int]  # if not present then pick one [1, 100]?

    # smart contract information
    contract_address: Optional[str]
    parameters: Optional[ExperimentParameters]

    # status information
    is_owner: bool = False


class ExperimentList(BaseListModel):
    """
    A paged list of experiments

    Attributes:

    * `items` - The list of experiments
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[Experiment]


class Status(BaseModel):
    """
    A summary of the experiment status (dynamic stuff!)

    Attributes:

    * `experiment` - The name of the experiment being run
    * `state` - The current state. It will be one of the following values:
      - `"unstarted"`
      - `"voting"`
      - `"training"`
      - `"waiting"`
      - `"stopped"`
    * `epoch` - The current epoch of the experiment
    """
    experiment: str
    state: str
    epoch: int

    @validator('state')
    def state_value_is_correct(cls, v):
        valid_states = ('unstarted', 'voting', 'training', 'waiting')
        if v not in valid_states:
            raise ValueError(f'state must be one of {",".join(valid_states)}')
        return v


class Performance(BaseModel):
    """
    An accuracy / performance data point

    Attributes:

    * `epoch` - The epoch of the data point
    * `performance` - The performance for this epoch
    """
    epoch: int
    performance: float


class PerformanceList(BaseListModel):
    """
    A paged list of performance data points

    Attributes:

    * `items` - The list of data points
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[Performance]


class Vote(BaseModel):
    """
    A voting snapshot

    Attribute:

    * `epoch` - The epoch this vote was cast
    * `vote` - The vote value
    * `is_proposer` - Flag to signal that the learner proposed this weight batch
    """
    epoch: int
    vote: bool
    is_proposer: bool


class VoteList(BaseListModel):
    """
    A paged list of votes

    Attributes:

    * `items` - The list of votes
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_last` - Flag to signal this is the last page of the results
    """
    items: List[Vote]


class Statistic(BaseModel):
    """
    A summary set of statistics

    Attributes:

    * `mean` - The mean value for the statistics

    Optional Attributes:

    * `minimum` - The minimum value
    * `maximum` - The maximum value
    * `stddev` - The estimate for the standard deviation
    """
    mean: float
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    stddev: Optional[float] = None


class Statistics(BaseModel):
    """
    The set of summary statistics for experiment

    Attributes:

    * `epoch_time` - The statistics for the epochs time (seconds)
    * `train_time` - The statistics for the model training time (seconds)
    * `evaluate_time` - The statistics for the model evaluation time (seconds)
    """
    epoch_time: Statistic
    train_time: Statistic
    evaluate_time: Statistic


class Empty(BaseModel):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------------------------------------------------------

@app.get("/")
def index():
    """
    Simple endpoint, useful for API health checking.
    """
    return {'state': 'alive and kicking!'}


@app.get('/datasets/', response_model=DatasetList, tags=['datasets'])
def get_list_datasets(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get the list of datasets that are present on this learner.

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.

    """
    return {}


@app.get('/datasets/{name}/', response_model=Dataset, tags=['datasets'])
def get_specific_dataset_information(name: str):
    """
    Lookup details on a specific dataset

    Route Parameters:

    * `name` - The name of the dataset to be looked up

    """
    return {}


@app.post('/datasets/', tags=['datasets'])
def create_new_dataset(dataset: Dataset):
    """
    Create / register a new dataset with the learner
    """
    return {}


@app.delete('/datasets/{name}/', tags=['datasets'])
def delete_dataset(name: str):
    """
    Delete the specified dataset

    Route parameters:

    * `name` - The name of the dataset to be deleted
    """
    return {}


@app.get('/models/', response_model=ModelList, tags=['models'])
def get_list_of_models(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get a list of the models that are present on the system

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return {}


@app.post('/models/', tags=['models'])
def create_new_model(model: Union[TrainedModel, Model]):
    """
    Create a new model
    """
    return {}


@app.get('/models/{name}/', tags=['models'])
def get_specific_model_information(name: str):
    """
    Lookup model information about a specific model

    Route Parameters:

    * `name` - The name of the model to be queried
    """
    return {}


@app.post('/models/{name}/', response_model=Model, tags=['models'])
def update_specific_model_information(name: str, model: Union[TrainedModel, Model]):
    """
    Update a specific model

    Route Parameters:

    * `name` - The name of the model to be updated
    """
    return {}


@app.delete('/models/{name}/', tags=['models'])
def delete_specific_model(name: str):
    """
    Delete a specific model

    Route Parameters:

    * `name` - The name of the model to be deleted
    """
    return {}


@app.post('/models/{name}/export/', response_model=TrainedModel, tags=['models'])
def export_model(name: str):
    """
    Export a trained model with the parameters and trained weights. This can be then consumed by a default machine
    learning pipeline

    Route Parameters:

    * `name` - The name of the model to be exported
    """
    return {}


@app.post('/models/{name}/copy/', tags=['models'])
def duplicate_model(name: str, params: CopyParams):
    """
    Create a copy of the specified model

    """
    return {}


@app.get('/node/info/', response_model=Info, tags=['node'])
def get_learner_information():
    """
    Get the static learner information. This is information that is not expected to change for the lifetime of the
    learner.
    """
    return {}


@app.get('/node/queue/active/', response_model=QueueList, tags=['node'])
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
    return {}


@app.get('/node/queue/pending/', response_model=QueueList, tags=['node'])
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
    return {}


@app.get('/experiments/', response_model=ExperimentList, tags=['experiments'])
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


@app.get('/experiments/{name}/', response_model=Experiment, tags=['experiments'])
def get_specific_experiment(name: str):
    """
    Lookup information about a specific experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@app.post('/experiments/{name}/', response_model=Experiment, tags=['experiments'])
def update_specific_experiment(name: str, experiment: Experiment):
    """
    Update the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """
    return {}


@app.delete('/experiments/{name}/', tags=['experiments'])
def delete_specific_experiment(name: str):
    """
    Delete the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated (`current` is an alias for the most recently started experiment)
    """
    return {}


@app.post('/experiments/', response_model=Experiment, tags=['experiments'])
def create_a_new_experiment():
    """
    Create a new experiment
    """
    return {}


@app.get('/experiments/{name}/status/', response_model=Status, tags=['experiments'])
def get_learner_status(name: str):
    """
    Get the status of the current experiment (if there is one). This information is expected to be updated frequently
    as the experiment proceeds.

    It is expected that clients will poll this API endpoint to collect up to date status information for the experiment

    Route Parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@app.get('/experiments/{name}/performance/{mode}/', response_model=PerformanceList, tags=['experiments'])
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


@app.get('/experiments/{name}/votes/', response_model=VoteList, tags=['experiments'])
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


@app.get('/experiments/{name}/stats/', response_model=Statistics, tags=['experiments'])
def get_learner_statistics(name: str):
    """
    Query basic statistics about the current experiment

    Route parameters:

    * `name` - The name of the experiment to be queried (`current` is an alias for the most recently started experiment)
    """
    return {}


@app.post('/experiments/{name}/start/', response_model=Empty, tags=['experiments'])
def start_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    return {}


@app.post('/experiments/{name}/join/', response_model=Empty, tags=['experiments'])
def join_an_experiment(name: str):
    """
    Starts the named experiment

    Route Parameters:

    * `name` - The name of the experiment to be started
    """
    return {}


@app.post('/experiments/{name}/leave/', response_model=Empty, tags=['experiments'])
def leave_the_current_experiment(name: str):
    """
    Leave the current experiment

    Route Parameters:

    * `name` - The name of the experiment to leave
    """
    return {}


@app.post('/experiments/{name}/stop/', response_model=Empty, tags=['experiments'])
def stop_the_current_experiment(name: str):
    """
    Stop the current experiment. This also prohibts other users from participating in the experiment

    Route Parameters:

    * `name` - The name of the experiment to stop
    """
    return {}
