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
