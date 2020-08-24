from typing import List

from colearn_interface.collective_learning_interface import Learner_Interface


def run(n_epochs: int, learners: List[Learner_Interface]):

    for i in range(n_epochs):
        proposer = i % len(learners)
        new_weights = learners[proposer].train_model()

        votes = [l.test_model(new_weights) for l in learners]
        approves = sum(1 if v.vote else 0 for v in votes)

        if approves >= len(learners) / 2:
            for l in learners:
                l.accept_weights(new_weights)
