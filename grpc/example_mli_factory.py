import json
from typing import Set, Dict

from colearn.ml_interface import MachineLearningInterface
from colearn_other.mli_factory import TaskType, mli_factory
from grpc.mli_factory_interface import MliFactory


# TODO Add Documentation
# TODO Add tests
class ExampleMliFactory(MliFactory):

    def __init__(self):
        self.models = set(str(task) for task in TaskType)
        self.dataloaders = set(str(task) for task in TaskType)
        self.compatibilities = {task: set(task) for task in TaskType}

    def get_models(self) -> Set[str]:
        return self.models

    def get_dataloaders(self) -> Set[str]:
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

        data_config = json.loads(dataset_params)

        train_folder = data_config["train_folder"]
        test_folder = data_config["test_folder"]

        model_config = json.loads(model_params)
        model_type = model_config["model_type"]

        entire_config = data_config.update(model_config)

        return mli_factory(str_task_type=model_name,
                           train_folder=train_folder,
                           str_model_type=model_type,
                           test_folder=test_folder,
                           **entire_config)