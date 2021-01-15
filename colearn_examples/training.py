from typing import List, Sequence

from colearn_examples.config import ColearnConfig, ModelConfig, TrainingData, TrainingMode
from colearn_examples.utils.results import Result, Results

from colearn.basic_learner import LearnerData
from colearn.ml_interface import ProposedWeights, MachineLearningInterface
from colearn.standalone_driver import run_one_epoch


def setup_models(config: ModelConfig, client_data_folders_list: List[str],
                 data_loading_func, test_data_dir=None):
    learner_datasets: List[LearnerData] = []
    n_learners = len(client_data_folders_list)
    for i in range(n_learners):
        learner_datasets.append(
            data_loading_func(config, client_data_folders_list[i], test_data_dir=test_data_dir)
        )

    all_learner_models = [config.model_type(config, data=learner_datasets[i]) for i in range(n_learners)]

    set_equal_weights(all_learner_models)

    return all_learner_models


def set_equal_weights(learners: Sequence[MachineLearningInterface]):
    first_learner_weights = learners[0].mli_get_current_weights()

    for learner in learners[1:]:
        learner.mli_accept_weights(first_learner_weights)


def initial_result(learners: Sequence[MachineLearningInterface]):
    result = Result()
    for learner in learners:
        proposed_weights = learner.mli_test_weights(learner.mli_get_current_weights())  # type: ProposedWeights
        result.test_scores.append(proposed_weights.test_score)
        result.vote_scores.append(proposed_weights.vote_score)
        result.votes.append(True)
    return result


def collective_learning_round(learners: Sequence[MachineLearningInterface], vote_threshold,
                              epoch):
    print("Doing collective learning round")
    result = Result()

    proposed_weights_list, vote = run_one_epoch(epoch, learners,
                                                vote_threshold)
    result.vote = vote
    result.votes = [pw.vote for pw in proposed_weights_list]
    result.vote_scores = [pw.vote_score for pw in
                          proposed_weights_list]
    result.test_scores = [pw.test_score for pw in proposed_weights_list]
    result.block_proposer = epoch % len(learners)

    return result


def individual_training_round(learners, epoch):
    print("Doing individual training pass")
    result = Result()

    # train all models
    for i, learner in enumerate(learners):
        print(f"Training learner #{i} epoch {epoch}")
        weights = learner.mli_propose_weights()
        proposed_weights = learner.mli_test_weights(weights)
        learner.mli_accept_weights(weights)

        result.votes.append(True)
        result.vote_scores.append(proposed_weights.vote_score)
        result.test_scores.append(proposed_weights.test_score)

    return result


def main(colearn_config: ColearnConfig, data_dir):
    results = Results()
    kwargs = {}

    # pylint: disable=C0415
    if colearn_config.data == TrainingData.XRAY:
        from colearn_examples.xray import (
            split_to_folders, display_statistics,
            plot_results, plot_votes, prepare_single_client,
            XrayConfig)
        model_config: ModelConfig = XrayConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.MNIST:
        # noinspection PyUnresolvedReferences
        from colearn_examples.mnist import (  # type: ignore [no-redef]
            split_to_folders, display_statistics,
            plot_results, plot_votes, prepare_single_client,
            MNISTConfig)
        model_config = MNISTConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.FRAUD:
        # noinspection PyUnresolvedReferences
        from colearn_examples.fraud import (  # type: ignore [no-redef]
            split_to_folders, display_statistics,
            plot_results, plot_votes, prepare_single_client,
            FraudConfig)
        model_config = FraudConfig(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.CIFAR10:
        # noinspection PyUnresolvedReferences
        from colearn_examples.cifar10 import (  # type: ignore [no-redef]
            split_to_folders, display_statistics,
            plot_results, plot_votes, prepare_single_client,
            CIFAR10Config)
        model_config = CIFAR10Config(colearn_config.shuffle_seed)
    elif colearn_config.data == TrainingData.COVID:
        # noinspection PyUnresolvedReferences
        from colearn_examples.covid_xray import (  # type: ignore [no-redef]
            split_to_folders, display_statistics,
            plot_results, plot_votes, prepare_single_client,
            CovidXrayConfig)
        model_config = CovidXrayConfig(colearn_config.shuffle_seed)
        kwargs["test_ratio"] = 0.2
    else:
        raise Exception("Unknown task: %s" % colearn_config.data)

    # load, shuffle, clean, and split the data into n_learners
    data_split = [1 / colearn_config.n_learners] * colearn_config.n_learners
    client_data_folders_list = split_to_folders(
        data_dir, colearn_config.shuffle_seed, data_split, colearn_config.n_learners, **kwargs
    )
    test_data_dir = None
    if colearn_config.data == TrainingData.COVID:
        test_data_dir = client_data_folders_list[-1]
        del client_data_folders_list[-1]

    # setup n_learners duplicate models before training
    all_learner_models = setup_models(
        model_config, client_data_folders_list, prepare_single_client, test_data_dir
    )  # type: List[MachineLearningInterface]

    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for i in range(colearn_config.n_epochs):
        if colearn_config.mode == TrainingMode.COLLECTIVE:
            results.data.append(
                collective_learning_round(all_learner_models,
                                          colearn_config.vote_threshold, i)
            )
        elif colearn_config.mode == TrainingMode.INDIVIDUAL:
            results.data.append(individual_training_round(all_learner_models, i))
        else:
            raise Exception("Unknown training mode")

        display_statistics(results, colearn_config.n_learners, colearn_config.mode,
                           colearn_config.vote_threshold, model_config, i + 1)

        if colearn_config.plot_results:
            # then make an updating graph
            plot_results(results, colearn_config.n_learners, block=False)
            if colearn_config.mode == TrainingMode.COLLECTIVE:
                plot_votes(results, block=False)

    if colearn_config.plot_results:
        if colearn_config.mode == TrainingMode.COLLECTIVE:
            plot_results(results, colearn_config.n_learners, block=False)
            plot_votes(results, block=True)
        else:
            plot_results(results, colearn_config.n_learners, block=True)
