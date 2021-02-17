import copy
import json
from typing import Set, Dict, Any

from colearn.ml_interface import MachineLearningInterface
from colearn_grpc.mli_factory_interface import MliFactory
from colearn_grpc.factory_registry import FactoryRegistry

# These are imported to they are registered in the FactoryRegistry and are available here
# pylint: disable=W0611
import colearn_keras.keras_mnist  # type:ignore # noqa: F401
import colearn_keras.keras_cifar10  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_xray  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_covid_xray  # type:ignore # noqa: F401
import colearn_other.fraud_dataset  # type:ignore # noqa: F401


# TODO Add Documentation
class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = {name: config.default_parameters for name, config
                       in FactoryRegistry.model_architectures.items()}
        self.dataloaders = {name: config.default_parameters for name, config
                            in FactoryRegistry.dataloaders.items()}

        self.compatibilities = {name: config.compatibilities for name, config
                                in FactoryRegistry.model_architectures.items()}

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return self.models

    def get_dataloaders(self) -> Dict[str, Dict[str, Any]]:
        return self.dataloaders

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

        data_config = copy.deepcopy(self.dataloaders[dataloader_name])  # Default parameters
        data_config.update(json.loads(dataset_params))

        # TODO Names should match between colearn and contract_learn
        data_config["train_folder"] = data_config["location"]
        prepare_data_loaders = FactoryRegistry.dataloaders[dataloader_name][0]
        data_loaders = prepare_data_loaders(**data_config)

        model_config = copy.deepcopy(self.models[model_name])  # Default parameters
        model_config.update(json.loads(model_params))
        prepare_learner = FactoryRegistry.model_architectures[model_name][0]

        return prepare_learner(data_loaders=data_loaders, **model_config)
