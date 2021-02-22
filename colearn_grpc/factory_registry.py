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
from inspect import signature
from typing import Callable, Dict, Any, List, NamedTuple


def _get_defaults(to_call: Callable) -> Dict[str, Any]:
    return {param.name: param.default
            for param in signature(to_call).parameters.values()
            if param.default != param.empty}


class FactoryRegistry:

    class DataloaderDef(NamedTuple):
        callable: Callable
        default_parameters: Dict[str, Any]
    dataloaders: Dict[str, DataloaderDef] = {}

    class ModelArchitectureDef(NamedTuple):
        callable: Callable
        default_parameters: Dict[str, Any]
        compatibilities: List[str]

    model_architectures: Dict[str, ModelArchitectureDef] = {}

    @classmethod
    def register_dataloader(cls, name: str):
        def wrap(dataloader: Callable):
            if name in cls.dataloaders:
                print(f"Warning: {name} already registered. Replacing with {dataloader.__name__}")
            cls.dataloaders[name] = cls.DataloaderDef(
                callable=dataloader,
                default_parameters=_get_defaults(dataloader))
            return dataloader
        return wrap

    @classmethod
    def register_model_architecture(cls, name: str, compatibilities: List[str]):
        def wrap(model_arch_creator: Callable):
            if name in cls.model_architectures:
                print(f"Warning: {name} already registered. Replacing with {model_arch_creator.__name__}")
            cls.model_architectures[name] = cls.ModelArchitectureDef(
                callable=model_arch_creator,
                default_parameters=_get_defaults(model_arch_creator),
                compatibilities=compatibilities)

            return model_arch_creator
        return wrap
