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
import json
import pytest

from colearn_keras.keras_mnist import split_to_folders
from colearn_keras.keras_learner import KerasLearner

# These are imported here so that they are registered in the FactoryRegistry
# pylint: disable=W0611
import colearn_keras.keras_mnist  # type:ignore # noqa: F401
import colearn_keras.keras_cifar10  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_xray  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_covid_xray  # type:ignore # noqa: F401
import colearn_other.fraud_dataset  # type:ignore # noqa: F401

from colearn_grpc.example_mli_factory import ExampleMliFactory

DATALOADER_NAMES = {"PYTORCH_XRAY", "KERAS_MNIST", "KERAS_CIFAR10", "PYTORCH_COVID_XRAY", "FRAUD"}
MODEL_NAMES = {"PYTORCH_XRAY", "KERAS_MNIST", "KERAS_MNIST_RESNET", "KERAS_CIFAR10", "PYTORCH_COVID_XRAY", "FRAUD"}


@pytest.fixture
def factory() -> ExampleMliFactory:
    """Returns an ExampleMLIFactory"""
    return ExampleMliFactory()


def test_setup(factory):
    assert len(factory.get_models()) > 0
    assert len(factory.get_dataloaders()) > 0
    assert len(factory.get_compatibilities()) > 0


def test_model_names(factory):
    for model in MODEL_NAMES:
        assert model in factory.get_models().keys()
    print(factory.get_models())


def test_dataloader_names(factory):
    for dl in DATALOADER_NAMES:
        assert dl in factory.get_dataloaders().keys()

    assert len(factory.get_dataloaders()["KERAS_MNIST"]) > 0


def test_compatibilities(factory):
    for model in MODEL_NAMES:
        assert model in factory.get_models().keys()
        for dl in factory.get_compatibilities()[model]:
            assert dl in DATALOADER_NAMES


@pytest.fixture()
def mnist_config():
    folders = split_to_folders(10)

    return {
        'model_name': "KERAS_MNIST",
        'dataloader_name': "KERAS_MNIST",
        'location': folders[0],
    }


def test_get_mnist(factory, mnist_config):
    model_params = json.dumps({"steps_per_epoch": 20})

    dataset_params = json.dumps(
        {'location': mnist_config['location'],
         })

    mli = factory.get_mli(
        model_name=mnist_config['model_name'],
        model_params=model_params,
        dataloader_name=mnist_config['dataloader_name'],
        dataset_params=dataset_params)

    assert isinstance(mli, KerasLearner)
    assert mli.model_fit_kwargs["steps_per_epoch"] == 20


def test_triple_mnist(factory, mnist_config):
    default_params = json.dumps({})

    dataset_params = json.dumps(
        {'location': mnist_config['location'],
         })

    mli = factory.get_mli(
        model_name=mnist_config['model_name'],
        model_params=default_params,
        dataloader_name=mnist_config['dataloader_name'],
        dataset_params=dataset_params)

    assert isinstance(mli, KerasLearner)
    default_steps = mli.model_fit_kwargs["steps_per_epoch"]

    model_params = json.dumps({"steps_per_epoch": 40})

    mli = factory.get_mli(
        model_name=mnist_config['model_name'],
        model_params=model_params,
        dataloader_name=mnist_config['dataloader_name'],
        dataset_params=dataset_params)

    assert isinstance(mli, KerasLearner)
    assert mli.model_fit_kwargs["steps_per_epoch"] == 40

    mli = factory.get_mli(
        model_name=mnist_config['model_name'],
        model_params=default_params,
        dataloader_name=mnist_config['dataloader_name'],
        dataset_params=dataset_params)

    assert isinstance(mli, KerasLearner)
    assert mli.model_fit_kwargs["steps_per_epoch"] == default_steps
