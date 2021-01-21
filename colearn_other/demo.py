from typing import Optional, Sequence

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import plot_results, plot_votes
from colearn.utils.results import Results
from colearn_other.mli_factory import TaskType, mli_factory


def main(str_task_type: str,
         n_learners: int = 5,
         n_epochs: int = 20,
         vote_threshold: float = 0.5,
         train_data_folder: Optional[str] = None,
         test_data_folder: Optional[str] = None,
         str_model_type: Optional[str] = None,
         **learning_kwargs):
    """

    :param str_task_type: String task type
    :param n_learners: Number of learners
    :param n_epochs: Number of epochs
    :param vote_threshold: Minimum fraction of positive votes to accept new weights
    :param train_data_folder: Path to training dataset
    :param test_data_folder: Optional path to testing dataset
    :param str_model_type: String model type
    :param learning_kwargs: Learning parameters
    """
    # Resolve task type
    task_type = TaskType[str_task_type]

    # Load correct split to folders function
    # pylint: disable=C0415
    if task_type == TaskType.PYTORCH_XRAY:
        from colearn_pytorch.pytorch_xray import split_to_folders, ModelType
        if "vote_on_accuracy" in learning_kwargs:
            if learning_kwargs["vote_on_accuracy"]:
                score_name = "auc"
            else:
                score_name = "loss"
        else:
            score_name = "auc"
    elif task_type == TaskType.KERAS_MNIST:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_mnist import split_to_folders, ModelType  # type: ignore[no-redef]
        score_name = "categorical_accuracy"
    elif task_type == TaskType.KERAS_CIFAR10:
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_cifar10 import split_to_folders, ModelType  # type: ignore[no-redef]
        score_name = "categorical_accuracy"
    elif task_type == TaskType.PYTORCH_COVID_XRAY:
        # noinspection PyUnresolvedReferences
        from colearn_pytorch.pytorch_covid_xray import split_to_folders, ModelType  # type: ignore[no-redef]
        if "vote_on_accuracy" in learning_kwargs:
            if learning_kwargs["vote_on_accuracy"]:
                score_name = "categorical_accuracy"
            else:
                score_name = "loss"
        else:
            score_name = "categorical_accuracy"
    elif task_type == TaskType.FRAUD:
        # noinspection PyUnresolvedReferences
        from colearn_other.fraud_dataset import split_to_folders, ModelType  # type: ignore [no-redef]
        score_name = "accuracy"
    else:
        raise Exception("Task %s not part of the TaskType enum" % type)

    # Replace with default model type if not specified
    if str_model_type is None:
        str_model_type = ModelType(1).name

    # lOAD DATA
    train_data_folders = split_to_folders(
        data_dir=train_data_folder or "",
        n_learners=n_learners,
        train=True,
        **learning_kwargs)

    test_data_folders: Sequence[Optional[str]]
    if test_data_folder is not None:
        test_data_folders = split_to_folders(
            data_dir=test_data_folder,
            n_learners=n_learners,
            train=False,
            **learning_kwargs
        )
    else:
        test_data_folders = [None] * n_learners

    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(mli_factory(str_task_type=str_task_type,
                                              str_model_type=str_model_type,
                                              train_folder=train_data_folders[i],
                                              test_folder=test_data_folders[i],
                                              **learning_kwargs
                                              ))

    set_equal_weights(all_learner_models)

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
