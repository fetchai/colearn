from typing import List

from colearn.ml_interface import MachineLearningInterface


def run(n_epochs: int, learners: List[MachineLearningInterface]):

    for i in range(n_epochs):
        proposer = i % len(learners)
        new_weights = learners[proposer].train_model()

        prop_weights_list = [ln.test_model(new_weights) for ln in learners]
        approves = sum(1 if v.vote else 0 for v in prop_weights_list)

        if approves >= len(learners) / 2:
            for j, l in enumerate(learners):
                l.accept_weights(prop_weights_list[j])
