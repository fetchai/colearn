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
import abc
from enum import Enum
from typing import Any, Optional

import onnx
from pydantic import BaseModel


class DiffPrivBudget(BaseModel):
    target_epsilon: float
    target_delta: float
    consumed_epsilon: float
    consumed_delta: float


class ErrorCodes(Enum):
    DP_BUDGET_EXCEEDED = 1


class TrainingSummary(BaseModel):
    dp_budget: Optional[DiffPrivBudget]
    error_code: Optional[ErrorCodes]


class Weights(BaseModel):
    weights: Any
    training_summary: Optional[TrainingSummary]


class DiffPrivConfig(BaseModel):
    target_epsilon: float
    target_delta: float
    max_grad_norm: float
    noise_multiplier: float


class ProposedWeights(BaseModel):
    weights: Weights
    vote_score: float
    test_score: float
    vote: Optional[bool]


class ModelFormat(Enum):
    PICKLE_WEIGHTS_ONLY = 1
    ONNX = 2


class ColearnModel(BaseModel):
    model_format: ModelFormat
    model_file: Optional[str]
    model: Optional[Any]


class PredictionRequest(BaseModel):
    name: str
    input_data: Any
    pred_dataloader_key: Optional[Any]


class Prediction(BaseModel):
    name: str
    prediction_data: Any


def deser_model(model: Any) -> onnx.ModelProto:
    """
    Helper function to recover a onnx model from its deserialized form
    """
    return onnx.load_model_from_string(model)


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

    @abc.abstractmethod
    def mli_get_current_model(self) -> ColearnModel:
        """
        Returns the current model
        """
        pass

    @abc.abstractmethod
    def mli_make_prediction(self, request: PredictionRequest) -> Prediction:
        """
        Make prediction using the current model.
        Does not change the current weights of the model.

        :param request: data to get the prediction for
        :returns: the prediction
        """
        pass
