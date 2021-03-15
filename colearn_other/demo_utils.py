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
from typing import Callable


def get_split_to_folders(dataloader_name: str) -> Callable:
    # pylint: disable=C0415
    if dataloader_name == "PYTORCH_XRAY":
        # noinspection PyUnresolvedReferences
        from colearn_pytorch.pytorch_xray import split_to_folders  # type: ignore[no-redef]

    elif dataloader_name == "KERAS_MNIST":
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_mnist import split_to_folders  # type: ignore[no-redef]

    elif dataloader_name == "KERAS_CIFAR10":
        # noinspection PyUnresolvedReferences
        from colearn_keras.keras_cifar10 import split_to_folders  # type: ignore[no-redef]

    elif dataloader_name == "PYTORCH_COVID_XRAY":
        # noinspection PyUnresolvedReferences
        from colearn_pytorch.pytorch_covid_xray import split_to_folders  # type: ignore[no-redef]

    elif dataloader_name == "FRAUD":
        # noinspection PyUnresolvedReferences
        from colearn_other.fraud_dataset import split_to_folders  # type: ignore[no-redef]
    else:
        raise NotImplementedError("Split not defined for dataloader %s" % dataloader_name)

    return split_to_folders


def get_score_name(model_name: str) -> str:
    if model_name == "PYTORCH_XRAY":
        score_name = "auc"
    elif model_name in ["KERAS_MNIST", "KERAS_MNIST_RESNET", "KERAS_CIFAR10", "PYTORCH_COVID_XRAY"]:
        score_name = "categorical_accuracy"
    elif model_name == "FRAUD":
        score_name = "accuracy"
    else:
        score_name = "loss"
    return score_name
