# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
from typing import List

import torch
from sklearn.metrics import roc_auc_score


def binary_accuracy_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Function to compute binary classification accuracy based on model output (in logits) and ground truth labels

    :param outputs: Tensor of model output in logits
    :param labels: Tensor of ground truth labels
    :return: Fraction of correct predictions
    """
    outputs = (torch.sigmoid(outputs) > 0.5).float()
    correct = (outputs == labels).sum().item()
    return correct / labels.shape[0]


def auc_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Function to compute area under curve based on model outputs (in logits) and ground truth labels

    :param outputs: Tensor of model outputs in logits
    :param labels: Tensor of ground truth labels
    :return: AUC score
    """
    predictions = torch.sigmoid(outputs)
    return roc_auc_score(labels.cpu().numpy().astype(int), predictions.cpu().numpy())


def categorical_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Function to compute accuracy based on model prediction and ground truth labels

    :param outputs: Tensor of model predictions
    :param labels: Tensor of ground truth labels
    :return: Fraction of correct predictions
    """
    outputs = torch.argmax(outputs, 1).int()
    correct = (outputs == labels).sum().item()
    return correct / labels.shape[0]


def prepare_data_split_list(data, n: int) -> List[int]:
    """
    Create list of sizes for splitting

    :param data: dataset
    :param n: number of equal parts
    :return: list of sizes
    """

    parts = [len(data) // n] * n
    if sum(parts) < len(data):
        parts[-1] += len(data) - sum(parts)
    return parts
