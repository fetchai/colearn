from typing import Sequence

from colearn.ml_interface import ProposedWeights, MachineLearningInterface
from colearn.standalone_driver import run_one_train_round
from colearn.utils.results import Result


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
                              train_round):
    print("Doing collective learning round")
    result = Result()

    proposed_weights_list, vote = run_one_train_round(train_round, learners,
                                                vote_threshold)
    result.vote = vote
    result.votes = [pw.vote for pw in proposed_weights_list]
    result.vote_scores = [pw.vote_score for pw in
                          proposed_weights_list]
    result.test_scores = [pw.test_score for pw in proposed_weights_list]
    result.block_proposer = train_round % len(learners)

    return result


def individual_training_round(learners: Sequence[MachineLearningInterface], train_round):
    print("Doing individual training pass")
    result = Result()

    # train all models
    for i, learner in enumerate(learners):
        print(f"Training learner #{i} train round {train_round}")
        weights = learner.mli_propose_weights()
        proposed_weights = learner.mli_test_weights(weights)
        learner.mli_accept_weights(weights)

        result.votes.append(True)
        result.vote_scores.append(proposed_weights.vote_score)
        result.test_scores.append(proposed_weights.test_score)

    return result
