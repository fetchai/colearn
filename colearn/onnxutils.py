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
from typing import Any

import onnxmltools
import sklearn
import tensorflow as tf
import torch
from tensorflow import keras

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
