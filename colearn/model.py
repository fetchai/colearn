from typing import List

from colearn.ml_interface import ProposedWeights, MachineLearningInterface
from colearn.config import Config
from examples.utils.data import LearnerData


class BasicLearner(MachineLearningInterface):
    def __init__(self, config: Config, data: LearnerData, model=None):
        self.vote_accuracy = 0

        self.data = data
        self.config = config

        self._model = model or self._get_model()
        self.print_summary()

    def print_summary(self):
        raise NotImplementedError

    def get_name(self) -> str:
        return "Basic Learner"

    def _get_model(self):
        raise NotImplementedError

    def accept_weights(self, proposed_weights: ProposedWeights):
        # overwrite model weights with accepted weights from across the network
        self._set_weights(proposed_weights.weights)

        # update stored performance metrics with metrics computed based on
        # those weights
        self.vote_accuracy = proposed_weights.validation_accuracy

    def train_model(self):
        old_weights = self.get_weights()

        train_accuracy = self._train_model()

        new_weights = self.get_weights()
        self._set_weights(old_weights)

        print("Train accuracy: ", train_accuracy)

        return new_weights

    def _set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def test_model(self, weights=None) -> ProposedWeights:
        """Tests the proposed weights and fills in the rest of the fields"""
        if weights is None:
            weights = self.get_weights()

        proposed_weights = ProposedWeights()
        proposed_weights.weights = weights
        proposed_weights.validation_accuracy = self._test_model(weights,
                                                                validate=True)
        proposed_weights.test_accuracy = self._test_model(weights,
                                                          validate=False)
        proposed_weights.vote = (
            proposed_weights.validation_accuracy >= self.vote_accuracy
        )

        return proposed_weights

    def _test_model(self, weights=None, validate=False):
        raise NotImplementedError

    def _train_model(self):
        raise NotImplementedError

    def stop_training(self):
        raise NotImplementedError

    def clone(self, data=None):
        raise NotImplementedError


def setup_models(config: Config, learner_datasets: List[LearnerData]):
    all_learner_models = []
    clone_model = None
    if config.clone_model:
        clone_model = config.model_type(config, data=learner_datasets[0])

    for i in range(config.n_learners):
        if not config.clone_model:
            model = config.model_type(config, data=learner_datasets[i])
        else:
            model = clone_model.clone(data=learner_datasets[i])

        all_learner_models.append(model)

    return all_learner_models
