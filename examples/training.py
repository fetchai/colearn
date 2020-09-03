from typing import List

from colearn.config import Config, TrainingMode
from colearn.ml_interface import ProposedWeights
from colearn.basic_learner import BasicLearner
from colearn.standalone_driver import run_one_epoch
from examples.utils.utils import Result, Results


def setup_models(config: Config, client_data_folders_list: List[str]):
    learner_datasets = []
    for i in range(config.n_learners):
        learner_datasets.append(
            config.dataset.prepare_single_client(config,
                                                 client_data_folders_list[i])
        )

    all_learner_models = []
    clone_model = config.model_type(config, data=learner_datasets[0])

    for i in range(config.n_learners):
        model = clone_model.clone(data=learner_datasets[i])

        all_learner_models.append(model)

    return all_learner_models


def initial_result(learners: List[BasicLearner]):
    result = Result()
    for learner in learners:
        proposed_weights = learner.test_model()  # type: ProposedWeights
        learner.accept_weights(proposed_weights)
        test_accuracy = proposed_weights.test_accuracy
        vote_accuracy = proposed_weights.validation_accuracy
        result.test_accuracies.append(test_accuracy)
        result.vote_accuracies.append(vote_accuracy)
        result.votes.append(True)
    return result


def collaborative_training_pass(learners: List[BasicLearner], vote_threshold,
                                epoch):
    print("Doing collaborative training pass")
    result = Result()

    proposed_weights_list, vote = run_one_epoch(epoch, learners,
                                                vote_threshold)
    result.vote = vote
    result.votes = [pw.vote for pw in proposed_weights_list]
    result.vote_accuracies = [pw.validation_accuracy for pw in
                              proposed_weights_list]
    result.test_accuracies = [pw.test_accuracy for pw in proposed_weights_list]
    result.block_proposer = epoch % len(learners)

    return result


def individual_training_pass(learners):
    print("Doing individual training pass")
    result = Result()

    # train all models
    for i, learner in enumerate(learners):
        print("Training learner #", i)
        weights = learner.train_model()
        proposed_weights = learner.test_model(weights)
        learner.accept_weights(proposed_weights)

        result.votes.append(True)
        result.vote_accuracies.append(proposed_weights.validation_accuracy)
        result.test_accuracies.append(proposed_weights.test_accuracy)

    return result


def main(config: Config):
    results = Results()

    # load, shuffle, clean, and split the data into n_learners
    client_data_folders_list = config.dataset.split_to_folders(
        config, config.main_data_dir
    )

    # setup n_learners duplicate models before training
    all_learner_models = setup_models(
        config, client_data_folders_list
    )  # type: List[BasicLearner]

    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for i in range(config.n_epochs):
        if config.mode == TrainingMode.COLLABORATIVE:
            results.data.append(
                collaborative_training_pass(all_learner_models,
                                            config.vote_threshold, i)
            )
        elif config.mode == TrainingMode.INDIVIDUAL:
            results.data.append(individual_training_pass(all_learner_models))
        else:
            raise Exception("Unknown training mode")

        config.dataset.display_statistics(results, config, i + 1)

        if config.plot_results:
            # then make an updating graph
            config.dataset.plot_results(results, config, block=False)
            if config.mode == TrainingMode.COLLABORATIVE:
                config.dataset.plot_votes(results, block=False)

    if config.plot_results:
        config.dataset.plot_results(results, config, block=False)
        if config.mode == TrainingMode.COLLABORATIVE:
            config.dataset.plot_votes(results, block=True)
