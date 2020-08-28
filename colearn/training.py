import os
from random import randint
from typing import List

from colearn.config import Config, TrainingMode
from colearn.ml_interface import ProposedWeights
from colearn.model import BasicLearner, setup_models
from colearn.utils import Result, Results

# Uncomment that to disable GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_random_model(learners: List[BasicLearner], epoch, config):
    n_learners = len(learners)

    if config.random_proposer:
        block_proposer = randint(0, n_learners - 1)
    else:
        block_proposer = epoch % n_learners

    print("current block proposer: " + str(block_proposer))

    weights = learners[block_proposer].train_model()

    return block_proposer, weights


def train_all_models(learners: List[BasicLearner], result: Result):
    result.vote_accuracies = [0] * len(learners)
    result.test_accuracies = [0] * len(learners)
    for i in range(len(learners)):
        print("Training learner #", i)
        weights = learners[i].train_model()
        proposed_weights = learners[i].test_model(weights)
        validation_accuracy = proposed_weights.validation_accuracy
        test_accuracy = proposed_weights.test_accuracy
        proposed_weights = ProposedWeights()
        proposed_weights.test_accuracy = test_accuracy
        proposed_weights.weights = weights
        proposed_weights.validation_accuracy = validation_accuracy
        proposed_weights.vote = True
        learners[i].accept_weights(proposed_weights)
        result.vote_accuracies[i] = validation_accuracy
        result.test_accuracies[i] = test_accuracy


def validate_proposed_weights(learner: List[BasicLearner], weights, result: Result):
    votes = []
    vote_accuracies = [0] * len(learner)
    test_accuracies = [0] * len(learner)
    n_learners = len(learner)

    for i in range(n_learners):
        proposed_weights = learner[i].test_model(weights)
        vote_accuracies[i] = proposed_weights.validation_accuracy
        test_accuracies[i] = proposed_weights.test_accuracy
        vote = proposed_weights.vote
        votes.append(vote)

    result.votes = votes
    result.test_accuracies = test_accuracies
    result.vote_accuracies = vote_accuracies


def update_result(learner: List[BasicLearner], result: Result):
    n_learners = len(learner)

    for i in range(n_learners):
        # Test
        proposed_weights = learner[i].test_model()
        test_accuracy = proposed_weights.test_accuracy
        vote_accuracy = proposed_weights.validation_accuracy
        result.test_accuracies.append(test_accuracy)
        result.vote_accuracies.append(vote_accuracy)


def majority_vote(
    votes,
    all_learner_models: List[BasicLearner],
    result: Result,
    weights,
    block_proposer,
    vote_threshold,
):
    n_learners = len(all_learner_models)

    positive_votes = sum(votes) - votes[block_proposer]
    print(
        "Fraction of positive votes: "
        + str(positive_votes)
        + " / "
        + str((n_learners - 1))
    )

    if positive_votes >= ((n_learners - 1) * vote_threshold):
        result.vote = True
        print("POSITIVE VOTE")

        # block accepted
        for i in range(n_learners):
            proposed_weights = ProposedWeights()
            proposed_weights.test_accuracy = result.test_accuracies[i]
            proposed_weights.weights = weights
            proposed_weights.validation_accuracy = result.vote_accuracies[i]
            proposed_weights.vote = result.votes[i]
            all_learner_models[i].accept_weights(proposed_weights)
    else:
        result.vote = False
        print("NEGATIVE VOTE")
    return


def collaborative_training_pass(learners: List[BasicLearner], config, epoch):
    print("Doing collaborative training pass")
    result = Result()

    # train once for one of the randomly picked models
    result.block_proposer, weights = train_random_model(learners, epoch, config)

    # validate weights on all other models
    validate_proposed_weights(learners, weights, result)

    # majority vote
    majority_vote(
        result.votes,
        learners,
        result,
        weights,
        result.block_proposer,
        config.vote_threshold,
    )

    return result


def individual_training_pass(learners):
    print("Doing individual training pass")
    result = Result()

    # train all models
    train_all_models(learners, result)

    return result


def main(config: Config):
    results = Results()

    # load, shuffle, clean, and split the data into n_learners
    client_data_folders_list = config.dataset.split_to_folders(
        config, config.main_data_dir
    )

    all_learner_data = []
    for i in range(config.n_learners):
        all_learner_data.append(
            config.dataset.prepare_single_client(config, client_data_folders_list[i])
        )

    # setup n_learners duplicate models before training
    all_learner_models = setup_models(
        config, all_learner_data
    )  # type: List[BasicLearner]

    # Get initial accuracy
    result = Result()

    update_result(all_learner_models, result)
    result.votes = [True] * config.n_learners
    results.data.append(result)

    for i in range(config.n_epochs):
        if config.mode == TrainingMode.COLLABORATIVE:
            results.data.append(
                collaborative_training_pass(all_learner_models, config, i)
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
