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
from typing import Sequence

from colearn.ml_interface import ProposedWeights, MachineLearningInterface
from colearn.standalone_driver import run_one_round
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
                              round_index):
    print("Doing collective learning round")
    result = Result()

    proposed_weights_list, vote = run_one_round(round_index, learners,
                                                vote_threshold)
    result.vote = vote
    result.votes = [pw.vote for pw in proposed_weights_list]
    result.vote_scores = [pw.vote_score for pw in
                          proposed_weights_list]
    result.test_scores = [pw.test_score for pw in proposed_weights_list]
    result.block_proposer = round_index % len(learners)

    return result


def individual_training_round(learners: Sequence[MachineLearningInterface], round_index):
    print("Doing individual training pass")
    result = Result()

    # train all models
    for i, learner in enumerate(learners):
        print(f"Training learner #{i} round index {round_index}")
        weights = learner.mli_propose_weights()
        proposed_weights = learner.mli_test_weights(weights)
        learner.mli_accept_weights(weights)

        result.votes.append(True)
        result.vote_scores.append(proposed_weights.vote_score)
        result.test_scores.append(proposed_weights.test_score)

    return result
