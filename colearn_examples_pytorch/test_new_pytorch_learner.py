from unittest.mock import Mock, create_autospec, MagicMock

import torch
from torch.nn.modules.loss import _Loss
import torch.utils.data

import pytest

from colearn.ml_interface import Weights
from new_pytorch_learner import NewPytorchLearner


def get_mock_model() -> Mock:
    model = Mock()
    model.evaluate.return_value = {"loss": 1,
                                   "accuracy": 3}
    model.get_weights.return_value = "all the weights"
    return model


def get_mock_dataloader() -> Mock:
    # dl = MagicMock()
    # dl.__len__ = Mock(return_value=100)
    dl = create_autospec(torch.utils.data.DataLoader)
    dl.batch_size = 10
    return dl


def get_mock_optimiser() -> Mock:
    return Mock()


def get_mock_criterion() -> Mock:
    crit = create_autospec(_Loss, instance=True)
    return crit


@pytest.fixture
def nkl():
    """Returns a NewKeraslearner"""
    model = get_mock_model()
    dl = get_mock_dataloader()
    opt = get_mock_optimiser()
    crit = get_mock_criterion()
    nkl = NewPytorchLearner(model=model, train_loader=dl, optimizer=opt, criterion=crit)
    return nkl


def test_vote(nkl):
    assert nkl.vote_score == get_mock_model().evaluate.return_value["loss"]

    assert nkl.vote(1.1) is False
    assert nkl.vote(1) is True
    assert nkl.vote(0.9) is True


def test_minimise_criterion(nkl):
    nkl.minimise_criterion = False

    assert nkl.vote(1.1) is True
    assert nkl.vote(1) is True
    assert nkl.vote(0.9) is False


def test_criterion(nkl):
    nkl.criterion = "accuracy"
    nkl.mli_accept_weights(Weights(weights="foo"))
    assert nkl.vote_score == get_mock_model().evaluate.return_value["accuracy"]


def test_propose_weights(nkl):
    weights = nkl.mli_propose_weights()
    assert type(weights) == Weights
    assert weights.weights == get_mock_model().get_weights.return_value


def test_get_current_weights(nkl):
    weights = nkl.mli_get_current_weights()
    assert type(weights) == Weights
    assert weights.weights == get_mock_model().get_weights.return_value
