import abc
from typing import Dict, Set

from colearn.ml_interface import MachineLearningInterface


class MliFactory(abc.ABC):
    """
    Interface a class must implement to be used as a factory by the GRPC Server
    """

    @abc.abstractmethod
    def get_models(self) -> Set[str]:
        """
        Returns the set of models this factory produces
        """
        pass

    @abc.abstractmethod
    def get_datasets(self) -> Set[str]:
        """
        Returns the set of datasets this factory produces
        """
        pass

    @abc.abstractmethod
    def get_compatibilities(self) -> Dict[str, Set[str]]:
        """
        A model is compatible with a dataset if they can be used together to
        construct a MachineLearningInterface with the get_MLI function.

        Returns a dictionary that defines which model is compatible
        with which datasets.
        """
        pass

    @abc.abstractmethod
    def get_mli(self,
                model_name: str, model_params: str,
                dataset_name: str, dataset_params: str) -> MachineLearningInterface:
        """
        @param model_name: name of a model, must be in the set return by get_models
        @param model_params: optional, user defined parameters for the model
        @param dataset_name: name of a dataset:
            - must be in the set returned by get_datasets
            - must be compatible with model_name as defined by get_compatibilities
        @param dataset_params: optional, user defined parameters for the dataset
        @return: Instance of MachineLearningInterface
        Constructs an object that implements MachineLearningInterface whose
        underlying model is model_name and dataset is dataset_name.
        """
        pass
