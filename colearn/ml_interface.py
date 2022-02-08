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
from enum import IntEnum
from typing import Any, Optional, Tuple

import onnx
import onnxmltools
import sklearn
import tensorflow as tf
import torch
from pydantic import BaseModel
from tensorflow import keras

MODEL_SAVE_LOCATION = "./saved_model_colearn"
MODEL_BACKUP_LOCATION = "./backup_model_colearn"

model_classes_keras = (tf.keras.Model, keras.Model, tf.estimator.Estimator)
model_classes_scipy = (torch.nn.Module)
model_classes_sklearn = (sklearn.base.ClassifierMixin)


def convert_model_to_onnx(model: Any):
    """
    Helper function to convert a ML model to onnx format
    """
    if isinstance(model, model_classes_keras):
        return onnxmltools.convert_keras(model)
    if isinstance(model, model_classes_sklearn):
        return onnxmltools.convert_sklearn(model)
    if 'xgboost' in model.__repr__():
        return onnxmltools.convert_sklearn(model)
    if isinstance(model, model_classes_scipy):
        raise Exception("Pytorch models not yet supported to onnx")
    else:
        raise Exception("Attempt to convert unsupported model to onnx: {model}")


class DiffPrivBudget(BaseModel):
    target_epsilon: float
    target_delta: float
    consumed_epsilon: float
    consumed_delta: float


class TrainingSummary(BaseModel):
    dp_budget: Optional[DiffPrivBudget]


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

class TestResponse(BaseModel):
    vote_score: float
    test_score: float
    vote: Optional[bool]


class ModelFormat(IntEnum):
    PICKLE_WEIGHTS_ONLY = 1
    NATIVE = 2
    ONNX = 3

class ColearnModel(BaseModel):
    model_format: ModelFormat
    # If the model resides in a file (use when model is empty)
    model_file: Optional[str]
    model: Optional[Any]


# Convenience function for ColearnModel - bytes are either in a file or in the class
def get_model_bytes(model: ColearnModel):

    file_bytes = model.model

    if file_bytes is None:
        if model.model_file is not None:
            try:
                file_bytes = open(model.model_file, 'rb').read()
            except FileNotFoundError as e:
                _logger.error(f"Found error when trying to load model file {model.model_file}: {e}")
                return None
        else:
            _logger.error(f"No model or model filename supplied by colearn model when testing!")
            return None

    return file_bytes


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
    def mli_get_model(self) -> ColearnModel:
        """
        Returns the current model
        """
        pass

    @abc.abstractmethod
    def mli_set_model(self, model: ColearnModel):
        """
        Sets the current model
        """
        pass

    @abc.abstractmethod
    def mli_test_model(self, model: ColearnModel) -> TestResponse:
        """
        Tests the current model
        """
        pass

    @abc.abstractmethod
    def mli_propose_model(self) -> Tuple[ColearnModel, TestResponse]:
        """
        Trains the model. Returns the model, and its performance. Does not change the current weights of the model.
        """
        pass
