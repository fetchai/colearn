import json
from enum import Enum
from typing import Optional, Set, Dict

from colearn.ml_interface import MachineLearningInterface
from colearn.mli_factory_interface import MliFactory


class TaskType(Enum):
    PYTORCH_XRAY = 1
    KERAS_MNIST = 2
    KERAS_CIFAR10 = 3
    PYTORCH_COVID_XRAY = 4
    FRAUD = 5

    def __str__(self):
        return str(self.value)


def mli_factory(str_task_type: str,
                train_folder: str,
                str_model_type: str,
                test_folder: Optional[str] = None,
                **learning_kwargs):
    # Resolve task type
    task_type = TaskType[str_task_type]

    # Load task
    # pylint: disable=C0415
    if task_type == TaskType.PYTORCH_XRAY:
        from colearn_pytorch.pytorch_xray import prepare_learner, prepare_data_loaders, ModelType
    elif task_type == TaskType.KERAS_MNIST:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_mnist import (  # type: ignore[no-redef]
            prepare_learner, prepare_data_loaders, ModelType)
    elif task_type == TaskType.KERAS_CIFAR10:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_cifar10 import (  # type: ignore[no-redef]
            prepare_learner, prepare_data_loaders, ModelType)
    elif task_type == TaskType.PYTORCH_COVID_XRAY:
        # noinspection PyUnresolvedReferences
        from colearn_pytorch.pytorch_covid_xray import (  # type: ignore[no-redef]
            prepare_learner, prepare_data_loaders, ModelType)
    elif task_type == TaskType.FRAUD:
        # noinspection PyUnresolvedReferences
        from colearn_other.fraud_dataset import (  # type: ignore [no-redef]
            prepare_learner, prepare_data_loaders, ModelType)
    else:
        raise Exception("Task %s not part of the TaskType enum" % type)

    # Resolve model type
    model_type = ModelType[str_model_type]

    learner_dataloaders = prepare_data_loaders(train_folder=train_folder,
                                               test_folder=test_folder,
                                               **learning_kwargs)

    learner = prepare_learner(model_type=model_type,
                              data_loaders=learner_dataloaders,
                              **learning_kwargs)
    return learner


# TODO Add Documentation
# TODO Add tests
class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = set(str(task) for task in TaskType)
        self.datasets = set(str(task) for task in TaskType)
        self.compatibilities = {task: set(task) for task in TaskType}

    def get_models(self) -> Set[str]:
        self.models

    def get_datasets(self) -> Set[str]:
        self.datasets

    def get_compatibilities(self) -> Dict[str, Set[str]]:
        self.compatibilities

    def get_mli(self, model_name: str, model_params: str, dataset_name: str,
                dataset_params: str) -> MachineLearningInterface:
        if model_name not in self.models:
            raise Exception(f"Model {model_name} is not a valid model. "
                            f"Available models are: {self.models}")
        if dataset_name not in self.datasets:
            raise Exception(f"Dataset {dataset_name} is not a valid dataset. "
                            f"Available datasets are: {self.datasets}")
        if dataset_name not in self.compatibilities[model_name]:
            raise Exception(f"Dataset {dataset_name} is not compatible with {model_name}."
                            f"Compatible datasets are: {self.compatibilities[model_name]}")

        data_config = json.loads(dataset_params)

        train_folder = data_config["train_folder"]
        test_folder = data_config["test_folder"]

        model_config = json.loads(model_params)
        model_type = model_config["model_type"]

        entire_config = data_config.update(model_config)

        return mli_factory(str_task_type=model_name,
                           train_folder=train_folder,
                           str_model_type=model_type,
                           test_folder=test_folder,
                           **entire_config)


