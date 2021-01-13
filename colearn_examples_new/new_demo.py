#!/usr/bin/env python
from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.results import Results
from colearn_examples.utils.plot import plot_results, plot_votes
from enum import Enum


class TaskType(Enum):
    PYTORCH_XRAY = 1


def main(str_task_type: str,
         train_data_folder: str,
         n_learners=5,
         n_epochs=20,
         vote_threshold=0.5,
         train_ratio=1.0,
         test_data_folder=None,
         str_model_type=None,
         **learning_kwargs):
    # Resolve task type
    task_type = TaskType[str_task_type]

    # Load task
    if task_type == TaskType.PYTORCH_XRAY:
        from colearn_pytorch.pytorch_xray import split_to_folders, prepare_model, prepare_learner, prepare_data_loader, \
            ModelType
    else:
        raise Exception("Task %s not part of the TaskType enum" % type)

    # Resolve model type
    if str_model_type is not None:
        model_type = ModelType[str_model_type]
    else:
        # Get first model if not specified
        model_type = list(ModelType)[0].name

    # lOAD DATA
    train_data_folders = split_to_folders(
        train_data_folder,
        n_learners=n_learners,
        **learning_kwargs)

    if test_data_folder is not None:
        test_data_folders = split_to_folders(
            test_data_folder,
            n_learners=n_learners,
            output_folder='/tmp/xray_test',
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
                prepare_data_loader(train_data_folders[i], train=True, train_ratio=train_ratio, **learning_kwargs))
            learner_test_dataloaders.append(
                prepare_data_loader(train_data_folders[i], train=False, train_ratio=train_ratio, **learning_kwargs))

    all_learner_models = []
    for i in range(n_learners):
        model = prepare_model(model_type)
        all_learner_models.append(prepare_learner(model=model,
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
