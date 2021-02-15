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
from unittest.mock import Mock, create_autospec

import pytest
import torch
import torch.utils.data
from torch.nn.modules.loss import _Loss

from colearn.ml_interface import Weights
from colearn_pytorch.pytorch_learner import PytorchLearner

# torch does not correctly type-hint its tensor class so pylint fails
MODEL_PARAMETERS = [torch.tensor([3, 3]), torch.tensor([4, 4])]  # pylint: disable=not-callable
MODEL_PARAMETERS2 = [torch.tensor([5, 5]), torch.tensor([6, 6])]  # pylint: disable=not-callable
BATCH_SIZE = 2
TRAIN_BATCHES = 1
TEST_BATCHES = 1
LOSS = 12


def get_mock_model() -> Mock:
    model = create_autospec(torch.nn.Module, instance=True, spec_set=True)
    model.parameters.return_value = [x.clone() for x in MODEL_PARAMETERS]
    model.to.return_value = model
    return model


def get_mock_dataloader() -> Mock:
    dl = create_autospec(torch.utils.data.DataLoader, instance=True)
    dl.__len__ = Mock(return_value=100)
    # pylint: disable=not-callable
    dl.__iter__.return_value = [(torch.tensor([0, 0]),
                                 torch.tensor([0])),
                                (torch.tensor([1, 1]),
                                 torch.tensor([1]))]
    dl.batch_size = BATCH_SIZE
    return dl


def get_mock_optimiser() -> Mock:
    return Mock()


def get_mock_criterion() -> Mock:
    crit = create_autospec(_Loss, instance=True)

    # pylint: disable=not-callable
    crit.return_value = torch.tensor(LOSS)
    crit.return_value.backward = Mock()  # type: ignore[assignment]

    return crit


@pytest.fixture
def nkl():
    """Returns a Pytorchlearner"""
    model = get_mock_model()
    dl = get_mock_dataloader()
    opt = get_mock_optimiser()
    crit = get_mock_criterion()
    nkl = PytorchLearner(model=model, train_loader=dl,
                         optimizer=opt, criterion=crit,
                         num_train_batches=1,
                         num_test_batches=1)
    return nkl


def test_setup(nkl):
    assert str(MODEL_PARAMETERS) == str(nkl.model.parameters())
    vote_score = LOSS / (TEST_BATCHES * BATCH_SIZE)
    assert nkl.vote_score == vote_score


def test_vote(nkl):
    vote_score = LOSS / (TEST_BATCHES * BATCH_SIZE)
    assert nkl.vote_score == vote_score

    assert nkl.minimise_criterion is True
    assert nkl.vote(vote_score + 0.1) is False
    assert nkl.vote(vote_score) is True
    assert nkl.vote(vote_score - 0.1) is True


def test_vote_minimise_criterion(nkl):
    vote_score = LOSS / (TEST_BATCHES * BATCH_SIZE)
    assert nkl.vote_score == vote_score

    nkl.minimise_criterion = False

    assert nkl.vote(vote_score + 0.1) is True
    assert nkl.vote(vote_score) is True
    assert nkl.vote(vote_score - 0.1) is False


def test_accept_weights(nkl):
    nkl.mli_accept_weights(Weights(weights=MODEL_PARAMETERS2))
    assert str(nkl.model.parameters()) == str(MODEL_PARAMETERS2)


def test_propose_weights(nkl):
    current_weights = nkl.model.parameters()
    proposed_weights = nkl.mli_propose_weights()
    assert isinstance(proposed_weights, Weights)
    # current weights should not change
    assert str(current_weights) == str(nkl.model.parameters())
    # proposed_weights should be different from current_weights, but I cannot
    # find a way to test this!


def test_get_current_weights(nkl):
    weights = nkl.mli_get_current_weights()
    assert isinstance(weights, Weights)
    assert str(weights.weights) == str(MODEL_PARAMETERS)
    assert str(weights.weights) == str(nkl.model.parameters())
