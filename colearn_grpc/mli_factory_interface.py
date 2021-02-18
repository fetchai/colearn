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
import abc
from typing import Dict, Set, Any

from colearn.ml_interface import MachineLearningInterface


class MliFactory(abc.ABC):
    """
    Interface a class must implement to be used as a factory by the GRPC Server
    """

    @abc.abstractmethod
    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the models this factory produces.
        The key is the name of the model and the values are their default parameters
        """
        pass

    @abc.abstractmethod
    def get_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the dataloaders this factory produces.
        The key is the name of the dataloader and the values are their default parameters
        """
        pass

    @abc.abstractmethod
    def get_compatibilities(self) -> Dict[str, Set[str]]:
        """
        A model is compatible with a dataloader if they can be used together to
        construct a MachineLearningInterface with the get_MLI function.

        Returns a dictionary that defines which model is compatible
        with which dataloader.
        """
        pass

    @abc.abstractmethod
    def get_mli(self,
                model_name: str, model_params: str,
                dataloader_name: str, dataset_params: str) -> MachineLearningInterface:
        """
        @param model_name: name of a model, must be in the set return by get_models
        @param model_params: user defined parameters for the model
        @param dataloader_name: name of a dataloader to be used:
            - must be in the set returned by get_dataloaders
            - must be compatible with model_name as defined by get_compatibilities
        @param dataset_params: user defined parameters for the dataset
        @return: Instance of MachineLearningInterface
        Constructs an object that implements MachineLearningInterface whose
        underlying model is model_name and dataset is loaded by dataloader_name.
        """
        pass
