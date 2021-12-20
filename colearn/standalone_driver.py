# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
from typing import List, Sequence

from colearn.ml_interface import MachineLearningInterface


def run(n_rounds: int, learners: List[MachineLearningInterface]):
    for i in range(n_rounds):
        run_one_round(i, learners)


def run_one_round(round_index: int, learners: Sequence[MachineLearningInterface],
                  vote_threshold=0.5):

    # Get weights from proposer
    proposer = round_index % len(learners)
    new_weights = learners[proposer].mli_propose_weights()

    prop_weights_list = [ln.mli_test_weights(new_weights) for ln in learners]

    # Invalidate vote on self since not allowed
    prop_weights_list[proposer].vote = None

    approves = sum(1 if v.vote else 0 for v in prop_weights_list)

    vote = False
    if approves >= (len(prop_weights_list) - 1) * vote_threshold:
        vote = True
        # Set all learners to new weights
        for learner in learners:
            learner.mli_accept_weights(new_weights)

    return prop_weights_list, vote
