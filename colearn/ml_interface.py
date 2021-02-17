import abc
from typing import Any

from pydantic import BaseModel


class Weights(BaseModel):
    weights: Any


class ProposedWeights(BaseModel):
    weights: Weights
    vote_score: float
    test_score: float
    vote: bool


class MachineLearningInterface(abc.ABC):
    @abc.abstractmethod
    def mli_propose_weights(self) -> Weights:
        """
        Trains the model. Returns new weights. Does not change the current weights of the model.
        """
        pass

    @abc.abstractmethod
    def mli_test_weights(self, weights: Weights) -> ProposedWeights:
        """
        Tests the proposed weights and fills in the rest of the fields
        """

    @abc.abstractmethod
    def mli_accept_weights(self, weights: Weights):
        """
        Updates the model with the proposed set of weights
        :param weights: The new weights
        """
        pass

    @abc.abstractmethod
    def mli_get_current_weights(self) -> Weights:
        """
        Returns the current weights of the model
        """
        pass
