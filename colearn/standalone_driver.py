from typing import List, Sequence

from colearn.ml_interface import MachineLearningInterface


def run(n_rounds: int, learners: List[MachineLearningInterface]):
    for i in range(n_rounds):
        run_one_round(i, learners)


def run_one_round(round_index: int, learners: Sequence[MachineLearningInterface],
                  vote_threshold=0.5):
    proposer = round_index % len(learners)
    new_weights = learners[proposer].mli_propose_weights()

    prop_weights_list = [ln.mli_test_weights(new_weights) for ln in learners]
    approves = sum(1 if v.vote else 0 for v in prop_weights_list)

    vote = False
    if approves >= len(learners) * vote_threshold:
        vote = True
        for j, learner in enumerate(learners):
            learner.mli_accept_weights(prop_weights_list[j].weights)

    return prop_weights_list, vote
