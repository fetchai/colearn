from typing import Optional, List, Dict, Any

from pydantic import BaseModel, validator


class BaseListModel(BaseModel):
    current_page: int
    total_pages: int
    is_start: bool
    is_last: bool


class ErrorResponse(BaseModel):
    detail: str


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


class BaseExperiment(BaseModel):
    name: str
    training_mode: str = 'collective'
    model: str
    dataset: str
    seed: Optional[int]  # if not present then pick one [1, 100]?


class CreateExperiment(BaseExperiment):
    mode: str
    contract_address: Optional[str]
    parameters: ExperimentParameters

    @validator('mode')
    def mode_is_correct(cls, v):
        valid_states = ('owner', 'follower')
        if v not in valid_states:
            raise ValueError(f'state must be one of {",".join(valid_states)}')
        return v

    @validator('contract_address')
    def contract_address_is_present(cls, v, values):
        if values['mode'] == 'owner' and v is not None:
            raise ValueError('contract_address should not be provided in owner mode')
        if values['mode'] == 'follower' and v is None:
            raise ValueError('contract_address should be provided when trying to join')

        return v


class UpdateExperiment(BaseModel):
    training_mode: Optional[str]
    model: Optional[str]
    dataset: Optional[str]
    seed: Optional[int]
    contract_address: Optional[str]
    parameters: Optional[ExperimentParameters]


class Experiment(BaseExperiment):
    """
    An experiment definition

    Attributes:

    * `name` - The name of the experiment
    * `training_mode` - The training mode for the experiment. Should be one of the following values:
        - `'collective'` (Default)
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
