# type: ignore [no-redef]
from typing import Callable, List

from colearn_examples.config import ColearnConfig, ModelConfig, TrainingData, TrainingMode
from colearn_examples.utils.results import Result, Results

from colearn.basic_learner import BasicLearner, LearnerData
from colearn.ml_interface import ProposedWeights
from colearn.standalone_driver import run_one_epoch


def setup_models(config: ModelConfig, client_data_folders_list: List[str],
                 data_loading_func: Callable[[ModelConfig, str], LearnerData]):
    learner_datasets = []
    n_learners = len(client_data_folders_list)
    for i in range(n_learners):
        learner_datasets.append(
            data_loading_func(config, client_data_folders_list[i])
        )

    all_learner_models = []
    clone_model = config.model_type(config, data=learner_datasets[0])

    for i in range(n_learners):
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


def main(colearn_config: ColearnConfig):
    results = Results()

    # pylint: disable=C0415
    if colearn_config.data == TrainingData.XRAY:
        from colearn_examples.xray import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, XrayConfig
        model_config = XrayConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.MNIST:
        from colearn_examples.mnist import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, MNISTConfig
        model_config = MNISTConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.FRAUD:
        from colearn_examples.fraud import split_to_folders, display_statistics, \
            plot_results, plot_votes, prepare_single_client, FraudConfig
        model_config = FraudConfig(colearn_config.shuffle_seed)
    else:
        raise Exception("Unknown task: %s" % colearn_config.data)

    # load, shuffle, clean, and split the data into n_learners
    client_data_folders_list = split_to_folders(
        colearn_config, colearn_config.data_dir
    )

    # setup n_learners duplicate models before training
    all_learner_models = setup_models(
        model_config, client_data_folders_list, prepare_single_client
    )  # type: List[BasicLearner]

    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for i in range(colearn_config.n_epochs):
        if colearn_config.mode == TrainingMode.COLLABORATIVE:
            results.data.append(
                collaborative_training_pass(all_learner_models,
                                            colearn_config.vote_threshold, i)
            )
        elif colearn_config.mode == TrainingMode.INDIVIDUAL:
            results.data.append(individual_training_pass(all_learner_models))
        else:
            raise Exception("Unknown training mode")

        display_statistics(results, colearn_config, model_config, i + 1)

        if colearn_config.plot_results:
            # then make an updating graph
            plot_results(results, colearn_config, block=False)
            if colearn_config.mode == TrainingMode.COLLABORATIVE:
                plot_votes(results, block=False)

    if colearn_config.plot_results:
        plot_results(results, colearn_config, block=False)
        if colearn_config.mode == TrainingMode.COLLABORATIVE:
            plot_votes(results, block=True)
