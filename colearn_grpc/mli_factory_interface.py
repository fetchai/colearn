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
import abc
from typing import Dict, Set, Any, Optional
import os.path
from pkg_resources import get_distribution, DistributionNotFound

from colearn.ml_interface import MachineLearningInterface


class MliFactory(abc.ABC):
    """
    Interface a class must implement to be used as a factory by the GRPC Server
    """
    _version = "0.0.0"

    # https://stackoverflow.com/questions/17583443
    try:
        _dist = get_distribution('colearn')
        # Normalize case for Windows systems
        dist_loc = os.path.normcase(_dist.location)
        here = os.path.normcase(__file__)
        if not here.startswith(os.path.join(dist_loc, 'colearn')):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        pass
    else:
        _version = _dist.version

    def get_version(self) -> str:
        """
        Returns the version of this library....
        """
        return self._version

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
    def get_prediction_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the prediction dataloaders this factory produces.
        The key is the name of the dataloader and the values are their default parameters
        """
        pass

    @abc.abstractmethod
    def get_data_compatibilities(self) -> Dict[str, Set[str]]:
        """
        A model is compatible with a dataloader if they can be used together to
        construct a MachineLearningInterface with the get_MLI function.

        Returns a dictionary that defines which model is compatible
        with which dataloader.
        """
        pass

    @abc.abstractmethod
    def get_pred_compatibilities(self) -> Dict[str, Set[str]]:
        """
        A model is compatible with a prediction dataloader if they can be used together to
        construct a MachineLearningInterface with the get_MLI function.

        Returns a dictionary that defines which model is compatible
        with which prediction dataloader.
        """
        pass

    @abc.abstractmethod
    def get_mli(self,
                model_name: str, model_params: str,
                dataloader_name: str, dataset_params: str,
                prediction_dataloader_name: Optional[str],
                prediction_dataset_params: Optional[str]) -> MachineLearningInterface:
        """
        @param model_name: name of a model, must be in the set return by get_models
        @param model_params: user defined parameters for the model
        @param dataloader_name: name of a dataloader to be used:
            - must be in the set returned by get_dataloaders
            - must be compatible with model_name as defined by get_data_compatibilities
        @param dataset_params: user defined parameters for the dataset
        @param prediction_dataloader_name: name of a prediction dataloader to be used:
            - must be in the set returned by get_prediction_dataloaders
            - must be compatible with model_name as defined by get_pred_compatibilities
        @param prediction_dataset_params: user defined parameters for the prediction and preprocessing
        @return: Instance of MachineLearningInterface
        Constructs an object that implements MachineLearningInterface whose
        underlying model is model_name and dataset is loaded by dataloader_name.
        """
        pass
