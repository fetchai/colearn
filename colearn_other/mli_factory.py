# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
from enum import Enum
from typing import Optional
from colearn.ml_interface import MachineLearningInterface


class TaskType(Enum):
    PYTORCH_XRAY = 1
    KERAS_MNIST = 2
    KERAS_CIFAR10 = 3
    PYTORCH_COVID_XRAY = 4
    FRAUD = 5


def mli_factory(str_task_type: str,
                train_folder: str,
                str_model_type: str,
                test_folder: Optional[str] = None,
                **learning_kwargs) -> MachineLearningInterface:
    """
    MachineLearningInterface factory
    :param str_task_type: String task type
    :param train_folder: Path to training set
    :param str_model_type: String model type
    :param test_folder: Optional path to test set
    :param learning_kwargs: Learning parameters to be passed to dataloader and model
    :return: Specific instance of MachineLearningInterface
    """
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
