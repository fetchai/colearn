from unittest.mock import Mock, create_autospec

import json
import pytest

# import torch
# import torch.utils.data
# from torch.nn.modules.loss import _Loss
#
# from colearn.ml_interface import Weights
# from colearn_pytorch.pytorch_learner import PytorchLearner
#
## torch does not correctly type-hint its tensor class so pylint fails
# MODEL_PARAMETERS = [torch.tensor([3, 3]), torch.tensor([4, 4])]  # pylint: disable=not-callable
# MODEL_PARAMETERS2 = [torch.tensor([5, 5]), torch.tensor([6, 6])]  # pylint: disable=not-callable
# BATCH_SIZE = 2
# TRAIN_BATCHES = 1
# TEST_BATCHES = 1
# LOSS = 12
#
#
# def get_mock_model() -> Mock:
#    model = create_autospec(torch.nn.Module, instance=True, spec_set=True)
#    model.parameters.return_value = [x.clone() for x in MODEL_PARAMETERS]
#    model.to.return_value = model
#    return model
#
#
# def get_mock_dataloader() -> Mock:
#    dl = create_autospec(torch.utils.data.DataLoader, instance=True)
#    dl.__len__ = Mock(return_value=100)
#    # pylint: disable=not-callable
#    dl.__iter__.return_value = [(torch.tensor([0, 0]),
#                                 torch.tensor([0])),
#                                (torch.tensor([1, 1]),
#                                 torch.tensor([1]))]
#    dl.batch_size = BATCH_SIZE
#    return dl
#
#
# def get_mock_optimiser() -> Mock:
#    return Mock()
#
#
# def get_mock_criterion() -> Mock:
#    crit = create_autospec(_Loss, instance=True)
#
#    # pylint: disable=not-callable
#    crit.return_value = torch.tensor(LOSS)
#    crit.return_value.backward = Mock()  # type: ignore[assignment]
#
#    return crit
from colearn_other.mli_factory import TaskType, mli_factory
from colearn_keras.keras_mnist import ModelType, split_to_folders
from colearn_keras.keras_learner import KerasLearner


from grpc.example_mli_factory import ExampleMliFactory


@pytest.fixture
def factory() -> ExampleMliFactory:
    """Returns an ExampleMLIFactory"""
    return ExampleMliFactory()


def test_setup(factory):
    assert len(factory.get_models()) > 0
    assert len(factory.get_dataloaders()) > 0
    assert len(factory.get_compatibilities()) > 0


def test_model_names(factory):
    print("Models: ", factory.get_models())
    print("Data  : ", factory.get_dataloaders())
    print("Compat: ", factory.get_compatibilities())
    for task in TaskType:
        assert task.name in factory.get_models()


def test_dataloader_names(factory):
    for task in TaskType:
        assert task.name in factory.get_dataloaders()


def test_compatibilities(factory):
    for task in TaskType:
        assert task.name in factory.get_models()
        assert task.name in factory.get_compatibilities()[task.name]


@pytest.fixture()
def mnist_config():

    folders = split_to_folders(10)

    return {
        'task_type': TaskType.KERAS_MNIST,
        'model_type': ModelType(1).name,
        'train_folder': folders[0],
        'test_folder': "",
    }

def test_get_mnist(factory, mnist_config):

    model_params = json.dumps({'model_type': mnist_config['model_type']})

    dataset_params = json.dumps(
        {'train_folder': mnist_config['train_folder'],
         'test_folder': mnist_config['test_folder']
         })

    mli = factory.get_mli(
        model_name=mnist_config['task_type'].name,
        model_params=model_params,
        dataloader_name=mnist_config['task_type'].name,
        dataset_params=dataset_params)

    assert type(mli) == KerasLearner
