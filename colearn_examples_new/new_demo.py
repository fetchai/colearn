from enum import Enum

from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


class TaskType(Enum):
    PYTORCH_XRAY = 1
    KERAS_MNIST = 2
    KERAS_CIFAR10 = 3
    PYTORCH_COVID_XRAY = 4
    FRAUD = 5


def main(str_task_type: str,
         n_learners=5,
         n_epochs=20,
         vote_threshold=0.5,
         train_ratio=0.8,
         train_data_folder: str = None,
         test_data_folder: str = None,
         str_model_type: str = None,
         **learning_kwargs):
    # Resolve task type
    task_type = TaskType[str_task_type]

    # Load task
    # pylint: disable=C0415
    if task_type == TaskType.PYTORCH_XRAY:
        from colearn_pytorch.pytorch_xray import split_to_folders, prepare_learner, prepare_data_loader, ModelType
    elif task_type == TaskType.KERAS_MNIST:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_mnist import (  # type: ignore [no-redef]
            split_to_folders, prepare_learner, prepare_data_loader, ModelType)
    elif task_type == TaskType.KERAS_CIFAR10:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_cifar10 import (  # type: ignore [no-redef]
            split_to_folders, prepare_learner, prepare_data_loader, ModelType)
    elif task_type == TaskType.PYTORCH_COVID_XRAY:
        # noinspection PyUnresolvedReferences
        from colearn_pytorch.pytorch_covid_xray import (  # type: ignore [no-redef]
            split_to_folders, prepare_learner, prepare_data_loader, ModelType)
    elif task_type == TaskType.FRAUD:
        # noinspection PyUnresolvedReferences
        from colearn_examples_new.fraud import (  # type: ignore [no-redef]
            split_to_folders, prepare_learner, prepare_data_loader, ModelType)
    else:
        raise Exception("Task %s not part of the TaskType enum" % type)

    # Resolve model type
    if str_model_type is not None:
        model_type = ModelType[str_model_type]
    else:
        # Get first model if not specified
        model_type = list(ModelType)[0]

    # lOAD DATA
    train_data_folders = split_to_folders(
        data_dir=train_data_folder,
        n_learners=n_learners,
        train=True,
        **learning_kwargs)

    if test_data_folder is not None:
        test_data_folders = split_to_folders(
            data_dir=test_data_folder,
            n_learners=n_learners,
            train=False,
            **learning_kwargs
        )

    learner_train_dataloaders = []
    learner_test_dataloaders = []

    for i in range(n_learners):
        if test_data_folder is not None:
            learner_train_dataloaders.append(
                prepare_data_loader(train_data_folders[i], train=True, train_ratio=train_ratio, **learning_kwargs))
            learner_test_dataloaders.append(
                prepare_data_loader(test_data_folders[i], train=True, train_ratio=train_ratio, **learning_kwargs))
        else:
            learner_train_dataloaders.append(
                prepare_data_loader(train_data_folders[i], train=True, **learning_kwargs))
            learner_test_dataloaders.append(
                prepare_data_loader(train_data_folders[i], train=False, **learning_kwargs))

    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(prepare_learner(model_type=model_type,
                                                  train_loader=learner_train_dataloaders[i],
                                                  test_loader=learner_test_dataloaders[i],
                                                  **learning_kwargs))

    set_equal_weights(all_learner_models)
    score_name = all_learner_models[0].score_name

    # Now we're ready to start collective learning
    # Get initial accuracy
    results = Results()
    results.data.append(initial_result(all_learner_models))

    for epoch in range(n_epochs):
        results.data.append(
            collective_learning_round(all_learner_models,
                                      vote_threshold, epoch)
        )

        plot_results(results, n_learners, score_name=score_name)
        plot_votes(results)

    plot_results(results, n_learners, score_name=score_name)
    plot_votes(results, block=True)
