import json
from inspect import signature
from typing import Set, Dict, Any, Callable

from colearn.ml_interface import MachineLearningInterface
from colearn_grpc.mli_factory_interface import MliFactory
from colearn_keras.keras_mnist import prepare_data_loaders, ModelType
from colearn_keras.keras_cifar10 import prepare_data_loaders, ModelType
from colearn_pytorch.pytorch_xray import prepare_data_loaders, ModelType
from colearn_pytorch.pytorch_covid_xray import prepare_data_loaders, ModelType
from colearn_other.fraud_dataset import prepare_data_loaders, ModelType
from colearn_other.mli_factory import TaskType, mli_factory

from colearn_grpc.factory_registry import FactoryRegistry


# TODO Add Documentation
class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = {name: params for name, (_, params, _) in FactoryRegistry.model_architectures.items()}
        self.dataloaders = {name: params for name, (_, params) in FactoryRegistry.dataloaders.items()}

        # TODO Currently only KERAS_MNIST(2DConv) is supported
        #import colearn_keras.keras_mnist as Keras_Mnist
        self.models["KERAS_MNIST"]["model_type"] = "CONV2D"

        self.compatibilities = {name: dataloader_names
                                for name, (_, _, dataloader_names)
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



        data_config = self.dataloaders[dataloader_name]  # Default parameters
        print("1", data_config)

        data_config.update(json.loads(dataset_params))
        print("2", data_config)

        # TODO Names should match between colearn and contract_learn
        data_config["train_folder"] = data_config["location"]
        print("3", data_config)

        prepare_data_loaders = FactoryRegistry.dataloaders[dataloader_name][0]
        data_loaders = prepare_data_loaders(**data_config)



        model_config = self.models[model_name]  # Default parameters
        model_config.update(json.loads(model_params))
        # model_type will allow you to choose different architectures for different tasks
        # eventually we will get rid of it, but for now only the first model_type is supported
        # and the name of the architecture is just the task
        model_type = model_config["model_type"]
        model_config.pop('model_type', None)  # Required because model_type is passed as argument as well
        prepare_learner = FactoryRegistry.model_architectures[model_name][0]
        return prepare_learner(model_type=model_type, data_loaders=data_loaders, **model_config)

        ## Join both configs into one big config
        #data_config.update(model_config)

        #return mli_factory(str_task_type=model_name,
        #                   train_folder=train_folder,
        #                   str_model_type=model_type,
        #                   **data_config)

    def get_registry(self):
        return FactoryRegistry
