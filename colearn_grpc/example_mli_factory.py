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
from inspect import signature
from typing import Set, Dict, Any

from colearn.ml_interface import MachineLearningInterface
from colearn_grpc.mli_factory_interface import MliFactory
from colearn_keras.keras_mnist import prepare_data_loaders, ModelType
from colearn_other.mli_factory import TaskType, mli_factory


# TODO Add Documentation
class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = {task.name: {} for task in TaskType}
        self.dataloaders = {task.name: {} for task in TaskType}

        # TODO Currently only KERAS_MNIST(2DConv) is supported
        self.models[TaskType.KERAS_MNIST.name] = {"model_type": ModelType(1).name}
        self.dataloaders[TaskType.KERAS_MNIST.name] = \
            {param.name: param.default
             for param in signature(prepare_data_loaders).parameters.values()
             if param.default != param.empty}

        self.compatibilities = {task.name: {task.name} for task in TaskType}

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return self.models

    def get_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        return self.dataloaders

    def get_compatibilities(self) -> Dict[str, Set[str]]:
        return self.compatibilities

    def get_mli(self, model_name: str, model_params: str, dataloader_name: str,
                dataset_params: str) -> MachineLearningInterface:
        if model_name not in self.models:
            raise Exception(f"Model {model_name} is not a valid model. "
                            f"Available models are: {self.models}")
        if dataloader_name not in self.dataloaders:
            raise Exception(f"Dataloader {dataloader_name} is not a valid dataloader. "
                            f"Available dataloaders are: {self.dataloaders}")
        if dataloader_name not in self.compatibilities[model_name]:
            raise Exception(f"Dataloader {dataloader_name} is not compatible with {model_name}."
                            f"Compatible dataloaders are: {self.compatibilities[model_name]}")

        data_config = self.dataloaders[dataloader_name]  # Default parameters
        data_config.update(json.loads(dataset_params))

        # TODO Names should match between colearn and contract_learn
        train_folder = data_config["location"]

        model_config = self.models[model_name]  # Default parameters
        model_config.update(json.loads(model_params))
        # model_type will allow you to choose different architectures for different tasks
        # eventually we will get rid of it, but for now only the first model_type is supported
        # and the name of the architecture is just the task
        model_type = model_config["model_type"]
        model_config.pop('model_type', None)  # Required because model_type is passed as argument as well

        # Join both configs into one big config
        data_config.update(model_config)

        return mli_factory(str_task_type=model_name,
                           train_folder=train_folder,
                           str_model_type=model_type,
                           **data_config)
