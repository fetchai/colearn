from enum import Enum
from typing import Optional


class TaskType(Enum):
    PYTORCH_XRAY = 1
    KERAS_MNIST = 2
    KERAS_CIFAR10 = 3
    PYTORCH_COVID_XRAY = 4
    FRAUD = 5


def mli_factory(str_task_type: str,
                train_folder: str,
                str_model_type: Optional[str] = None,
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
    if str_model_type is not None:
        model_type = ModelType[str_model_type]
    else:
        # Get first model if not specified
        model_type = list(ModelType)[0]

    learner_dataloaders = prepare_data_loaders(train_folder=train_folder,
                                               test_folder=test_folder,
                                               **learning_kwargs)

    learner = prepare_learner(model_type=model_type,
                              data_loaders=learner_dataloaders,
                              **learning_kwargs)
    return learner
