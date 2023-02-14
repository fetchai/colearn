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
from inspect import signature
from typing import Callable, Dict, Any, List, NamedTuple


class RegistryException(Exception):
    pass


def _get_defaults(to_call: Callable) -> Dict[str, Any]:
    return {param.name: param.default
            for param in signature(to_call).parameters.values()
            if param.default != param.empty}


def check_dataloader_callable(to_call: Callable):
    sig = signature(to_call)
    if "location" not in sig.parameters:
        raise RegistryException("dataloader must accept a 'location' parameter")


class FactoryRegistry:
    class DataloaderDef(NamedTuple):
        callable: Callable
        default_parameters: Dict[str, Any]

    dataloaders: Dict[str, DataloaderDef] = {}

    class PredictionDataloaderDef(NamedTuple):
        callable: Callable
        default_parameters: Dict[str, Any]

    prediction_dataloaders: Dict[str, PredictionDataloaderDef] = {}

    class ModelArchitectureDef(NamedTuple):
        callable: Callable
        default_parameters: Dict[str, Any]
        data_compatibilities: List[str]
        pred_compatibilities: List[str]

    model_architectures: Dict[str, ModelArchitectureDef] = {}

    @classmethod
    def register_dataloader(cls, name: str):
        def wrap(dataloader: Callable):
            check_dataloader_callable(dataloader)
            if name in cls.dataloaders:
                print(
                    f"Warning: {name} already registered. Replacing with {dataloader.__name__}")
            cls.dataloaders[name] = cls.DataloaderDef(
                callable=dataloader,
                default_parameters=_get_defaults(dataloader))
            return dataloader

        return wrap

    @classmethod
    def register_prediction_dataloader(cls, name: str):
        def wrap(prediction_dataloader: Callable):
            check_dataloader_callable(prediction_dataloader)
            if name in cls.prediction_dataloaders:
                print(
                    f"Warning: {name} already registered. Replacing with {prediction_dataloader.__name__}")
            cls.prediction_dataloaders[name] = cls.PredictionDataloaderDef(
                callable=prediction_dataloader,
                default_parameters=_get_defaults(prediction_dataloader))
            return prediction_dataloader

        return wrap

    @classmethod
    def register_model_architecture(cls, name: str,
                                    data_compatibilities: List[str],
                                    pred_compatibilities: List[str]):
        def wrap(model_arch_creator: Callable):
            cls.check_model_data_callable(model_arch_creator, data_compatibilities)
            cls.check_model_prediction_callable(
                model_arch_creator, pred_compatibilities)
            if name in cls.model_architectures:
                print(
                    f"Warning: {name} already registered. Replacing with {model_arch_creator.__name__}")
            cls.model_architectures[name] = cls.ModelArchitectureDef(
                callable=model_arch_creator,
                default_parameters=_get_defaults(model_arch_creator),
                data_compatibilities=data_compatibilities,
                pred_compatibilities=pred_compatibilities)

            return model_arch_creator

        return wrap

    @classmethod
    def check_model_data_callable(cls, to_call: Callable, compatibilities: List[str]):
        sig = signature(to_call)
        if "data_loaders" not in sig.parameters:
            raise RegistryException("model must accept a 'data_loaders' parameter")
        model_dl_type = sig.parameters["data_loaders"].annotation
        for dl in compatibilities:
            if dl not in cls.dataloaders:
                raise RegistryException(f"Compatible dataloader {dl} is not registered. The dataloader needs to be "
                                        "registered before the model that references it.")
            dl_type = signature(cls.dataloaders[dl].callable).return_annotation
            if dl_type != model_dl_type:
                raise RegistryException(f"Compatible dataloader {dl} has return type {dl_type}"
                                        f" but model data_loaders expects type {model_dl_type}")

    @classmethod
    def check_model_prediction_callable(cls, to_call: Callable, compatibilities: List[str]):
        sig = signature(to_call)
        if "prediction_data_loaders" not in sig.parameters:
            raise RegistryException(
                "model must accept a 'prediction_data_loaders' parameter")
        model_dl_type = sig.parameters["prediction_data_loaders"].annotation
        for dl in compatibilities:
            if dl not in cls.prediction_dataloaders:
                raise RegistryException(f"Compatible prediction dataloader {dl} is not registered."
                                        "The dataloader needs to be "
                                        "registered before the model that references it.")
            # this is empty, todo fix error here
            dl_type = signature(
                cls.prediction_dataloaders[dl].callable).return_annotation  # this is empty
            print(dl_type)
            print(model_dl_type)
            if dl_type != model_dl_type:
                raise RegistryException(f"Compatible prediction dataloader {dl} has return type {dl_type}"
                                        f" but model prediction_data_loaders expects type {model_dl_type}")
