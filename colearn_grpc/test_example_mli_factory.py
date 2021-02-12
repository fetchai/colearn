import json
import pytest

from colearn_other.mli_factory import TaskType
from colearn_keras.keras_mnist import ModelType, split_to_folders
from colearn_keras.keras_learner import KerasLearner


from colearn_grpc.example_mli_factory import ExampleMliFactory


@pytest.fixture
def factory() -> ExampleMliFactory:
    """Returns an ExampleMLIFactory"""
    return ExampleMliFactory()


def test_setup(factory):
    assert len(factory.get_models()) > 0
    assert len(factory.get_dataloaders()) > 0
    assert len(factory.get_compatibilities()) > 0


def test_model_names(factory):
    for task in TaskType:
        assert task.name in factory.get_models().keys()
    print(factory.get_models())


def test_dataloader_names(factory):
    for task in TaskType:
        assert task.name in factory.get_dataloaders().keys()

    assert len(factory.get_dataloaders()[TaskType.KERAS_MNIST.name]) > 0


def test_compatibilities(factory):
    for task in TaskType:
        assert task.name in factory.get_models().keys()
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
        {'location': mnist_config['train_folder'],
         'test_folder': mnist_config['test_folder']
         })

    mli = factory.get_mli(
        model_name=mnist_config['task_type'].name,
        model_params=model_params,
        dataloader_name=mnist_config['task_type'].name,
        dataset_params=dataset_params)

    assert isinstance(mli, KerasLearner)
