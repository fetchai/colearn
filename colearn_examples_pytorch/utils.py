import torch
from sklearn.metrics import roc_auc_score


def binary_accuracy_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    outputs = (torch.sigmoid(outputs) > 0.5).float()
    correct = (outputs == labels).sum().item()
    return correct / labels.shape[0]


def auc_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.sigmoid(outputs)
    return roc_auc_score(labels.numpy().astype(int), predictions.numpy())


def categorical_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Function to compute accuracy based on model prediction and ground truth labels

    :param outputs: Tensor with batch of model preditions
    :param labels: Tensor with batch of ground truth labels
    :return: Number of correct predictions
    """
    outputs = torch.argmax(outputs, 1).int()
    correct = (outputs == labels).sum().item()
    return correct / labels.shape[0]
