from typing import List, Sequence

from colearn.ml_interface import MachineLearningInterface


def run(n_epochs: int, learners: List[MachineLearningInterface]):
    for i in range(n_epochs):
        run_one_epoch(i, learners)


def run_one_epoch(epoch_index: int, learners: Sequence[MachineLearningInterface],
                  vote_threshold=0.5):
    proposer = epoch_index % len(learners)
    new_weights = learners[proposer].propose_weights()

    prop_weights_list = [ln.test_weights(new_weights) for ln in learners]
    approves = sum(1 if v.vote else 0 for v in prop_weights_list)

    vote = False
    if approves >= len(learners) * vote_threshold:
        vote = True
        for j, learner in enumerate(learners):
            learner.accept_weights(prop_weights_list[j].weights)

    return prop_weights_list, vote
