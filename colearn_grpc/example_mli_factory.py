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
import copy
import json
from typing import Set, Dict, Any

from colearn.ml_interface import MachineLearningInterface
from colearn_grpc.mli_factory_interface import MliFactory
from colearn_grpc.factory_registry import FactoryRegistry
from colearn_grpc.logging import get_logger

_logger = get_logger(__name__)


class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = {name: config.default_parameters for name, config
                       in FactoryRegistry.model_architectures.items()}
        self.dataloaders = {name: config.default_parameters for name, config
                            in FactoryRegistry.dataloaders.items()}

        self.compatibilities = {name: config.compatibilities for name, config
                                in FactoryRegistry.model_architectures.items()}

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.models)

    def get_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.dataloaders)

    def get_compatibilities(self) -> Dict[str, Set[str]]:
        return self.compatibilities

    def get_mli(self, model_name: str, model_params: str, dataloader_name: str,
                dataset_params: str) -> MachineLearningInterface:

        print("Call to get_mli")
        print(f"model_name {model_name} -> params: {model_params}")
        print(f"dataloader_name {dataloader_name} -> params: {dataset_params}")

        if model_name not in self.models:
            raise Exception(f"Model {model_name} is not a valid model. "
                            f"Available models are: {self.models}")
        if dataloader_name not in self.dataloaders:
            raise Exception(f"Dataloader {dataloader_name} is not a valid dataloader. "
                            f"Available dataloaders are: {self.dataloaders}")
        if dataloader_name not in self.compatibilities[model_name]:
            raise Exception(f"Dataloader {dataloader_name} is not compatible with {model_name}."
                            f"Compatible dataloaders are: {self.compatibilities[model_name]}")

        dataloader_config = copy.deepcopy(self.dataloaders[dataloader_name])  # Default parameters
        dataloader_new_config = json.loads(dataset_params)
        for key in dataloader_new_config.keys():
            if key in dataloader_config or key == "location":
                dataloader_config[key] = dataloader_new_config[key]
            else:
                _logger.warning(f"Key {key} was included in the dataloader params but this dataloader "
                                f"({dataloader_name}) does not accept it.")

        prepare_data_loaders = FactoryRegistry.dataloaders[dataloader_name][0]
        data_loaders = prepare_data_loaders(**dataloader_config)

        model_config = copy.deepcopy(self.models[model_name])  # Default parameters
        model_new_config = json.loads(model_params)
        for key in model_new_config.keys():
            if key in model_config:
                model_config[key] = model_new_config[key]
            else:
                _logger.warning(f"Key {key} was included in the model params but this model ({model_name}) does not "
                                "accept it.")

        prepare_learner = FactoryRegistry.model_architectures[model_name][0]

        return prepare_learner(data_loaders=data_loaders, **model_config)
