import abc
from typing import Optional, Any, Dict
from pydantic import BaseModel


class Weights(BaseModel):
    weights: Any


class ProposedWeights(BaseModel):
    weights: Weights
    vote_score: float
    test_score: float
    vote: bool
    evaluation_results: Optional[Dict] = None


class MachineLearningInterface(abc.ABC):
    @abc.abstractmethod
    def mli_propose_weights(self) -> Weights:
        """
        Trains the model. This function may block. Returns new weights.
        """
        pass

    @abc.abstractmethod
    def mli_test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        """
        Tests the proposed weights and fills in the rest of the fields
        Also evaluate the model using the metrics specified in eval_config:
            eval_config = {
                    "name": lambda y_true, y_pred
                }
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
