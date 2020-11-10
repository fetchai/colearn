from typing import Optional, List, Dict, Any

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

class PagedModel(BaseModel):
    current_page: int
    total_pages: int
    is_start: bool
    is_last: bool


class Dataset(BaseModel):
    """
    The information to define a dataset that can be trained against

    Attributes:

    * `name` - The name of the dataset
    * `location` - The path to the dataset
    * `train_size` - The proportion of the dataset to be used for training. (0.0, 1.0)
    * `validation_size` - The proportion of the dataset to be used for validation. (0.0, 1.0)
    * `test_size` - The proportion of the dataset to be used for testing. (0.0, 1.0)
    """
    name: str
    location: str
    train_size: float
    validation_size: float
    test_size: float


class DatasetPage(PagedModel):
    """
    A paged list of datasets

    Attributes:

    * `items` - The list of datasets for this page
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
    """
    items: List[Dataset]


class Model(BaseModel):
    """
    A model definition

    Attributes:

    * `name` - The name of the model
    * `parameters` - The dictionary of parameters which configure the model
    """
    name: str
    parameters: Dict[str, Any]


class TrainedModel(Model):
    """
    A trained model definition

    Attributes:

    * `name` - The name of the model
    * `parameters` - The dictionary of parameters which configure the model
    * `weights` - A dictionary of weights corresponding to the model
    """
    weights: Dict[str, Any]


class ModelPage(PagedModel):
    """
    A paged list of models

    Attributes:

    * `items` - The list of models for this page
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
    """
    items: List[Model]


class Checkpoint(BaseModel):
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


class QueuePage(PagedModel):
    """
    A paged list of queued experiments

    Attributes:

    * `items` - The list of experiment names
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
    """
    items: List[str]


class Experiment(BaseModel):
    """
    An experiment definition

    Attributes:

    * `name` - The name of the experiment
    * `model` - The name of the model to be used
    * `dataset` - The name of the dataset to be used
    """
    name: str
    model: str
    dataset: str


class ExperimentPage(PagedModel):
    """
    A paged list of experiments

    Attributes:

    * `items` - The list of experiments
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
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


class PerformancePage(PagedModel):
    """
    A paged list of performance data points

    Attributes:

    * `items` - The list of data points
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
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


class VotePage(PagedModel):
    """
    A paged list of votes

    Attributes:

    * `items` - The list of votes
    * `current_page` - The current page index of the results
    * `total_pages` - The total number of pages for this query
    * `is_start` - Flag to signal this is the first page of the results
    * `is_end` - Flag to signal this is the last page of the results
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


@app.get('/datasets/', response_model=DatasetPage, tags=['datasets'])
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


@app.post('/datasets/{name}/', response_model=Dataset, tags=['datasets'])
def update_specific_dataset_information(name: str, dataset: Dataset):
    """
    Update a specific named dataset with updated information

    Route Parameters:

    * `name` - The name of the dataset to be queried

    """
    return {}


@app.post('/datasets/', tags=['datasets'])
def create_new_dataset(dataset: Dataset):
    """
    Create / register a new dataset with the learner
    """
    return {}


@app.get('/models/', response_model=ModelPage, tags=['models'])
def get_list_of_models(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get a list of the models that are present on the system

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
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
def update_specific_model_information(name: str, model: Model):
    """
    Update a specific model

    Route Parameters:

    * `name` - The name of the model to be updated
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


@app.post('/models/{name}/checkpoint/', tags=['models'])
def duplicate_model(name: str, checkpoint: Checkpoint):
    """
    Create a copy of the specified model

    """
    return {}


@app.post('/models/', tags=['models'])
def create_new_model(model: Model):
    """
    Create a new model
    """
    return {}


@app.get('/learner/info/', response_model=Info, tags=['learner'])
def get_learner_information():
    """
    Get the static learner information. This is information that is not expected to change for the lifetime of the
    learner.
    """
    return {}


@app.get('/learner/queue/active/', response_model=QueuePage, tags=['learner'])
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


@app.get('/learner/queue/pending/', response_model=QueuePage, tags=['learner'])
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


@app.get('/experiments/', response_model=ExperimentPage, tags=['experiments'])
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

    * `name` - The name of the experiment to be queried
    """
    return {}


@app.post('/experiment/{name}/', response_model=Experiment, tags=['experiments'])
def update_specific_experiment(name: str, experiment: Experiment):
    """
    Update the specified experiment

    Route Parameters:

    * `name` - The name of the experiment to be updated
    """
    return {}


@app.post('/experiments/', response_model=Experiment, tags=['experiments'])
def create_a_new_experiment():
    """
    Create a new experiment
    """
    return {}


@app.get('/experiments/current/status/', response_model=Status, tags=['experiments'])
def get_learner_status():
    """
    Get the status of the current experiment (if there is one). This information is expected to be updated frequently
    as the experiment proceeds.

    It is expected that clients will poll this API endpoint to collect up to date status information for the experiment
    """
    return {}


@app.get('/experiments/current/performance/{mode}/', response_model=PerformancePage, tags=['experiments'])
def get_performance(mode: str = Path(..., regex=r'(?:validation|test)'), start: Optional[int] = None,
                    end: Optional[int] = None, page: Optional[int] = None,
                    page_size: Optional[int] = None):
    """
    Queries the performance for a specific mode. This query can provide both current and historical information on the
    currently evaluated model

    The mode is expected to be one of the following:

    * `validation`
    * `test`

    Optional parameters:

    * `start` - the starting epoch number to return values from
    * `end` - the last epoch number to return values from
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return {}


@app.get('/experiments/current/votes/', response_model=VotePage, tags=['experiments'])
def get_vote_information(start: Optional[int] = None, end: Optional[int] = None, page: Optional[int] = None,
                         page_size: Optional[int] = None):
    """
    Queries the vote information, both present and historical.

    Optional parameters:

    * `start` - the starting epoch number to return values from
    * `end` - the last epoch number to return values from
    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.

    """
    return {}


@app.get('/experiments/current/stats/', response_model=Statistics, tags=['experiments'])
def get_learner_statistics():
    """
    Query basic statistics about the current experiment
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


@app.post('/experiments/current/stop/', response_model=Empty, tags=['experiments'])
def stop_the_current_experiment():
    """
    Stop the current experiment
    """
    return {}
