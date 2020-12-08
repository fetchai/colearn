import abc
from typing import Optional


class Weights:
    __slots__ = ('weights', )

    def __init__(self, weights):
        self.weights = weights


class ProposedWeights:
    __slots__ = ('weights', 'vote_accuracy', 'test_accuracy', 'vote', 'evaluation_results')

    def __init__(self):
        self.weights = None
        self.vote_accuracy = 0
        self.test_accuracy = 0
        self.vote = False
        self.evaluation_results = {}


class MachineLearningInterface(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the MLInterface
        """
        pass

    @abc.abstractmethod
    def train_model(self) -> Weights:
        """
        Trains the model. This function may block. Returns new weights.
        """
        pass

    @abc.abstractmethod
    def stop_training(self):
        """
        Stops the training of the model.
        """
        pass

    @abc.abstractmethod
    def clone(self):
        """
        Returns a copy of the MLInterface
        """
        pass

    @abc.abstractmethod
    def test_model(self, weights: Weights = None, eval_config: Optional[dict] = None) -> ProposedWeights:
        """
        Tests the proposed weights and fills in the rest of the fields
        Also evaluate the model using the metrics specified in eval_config:
            eval_config = {
                    "name": lambda y_true, y_pred
                }
        """

    @abc.abstractmethod
    def accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        pass
