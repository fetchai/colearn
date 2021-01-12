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


def prepare_data_split_list(data, n):
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
