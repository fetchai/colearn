from inspect import signature
from typing import Callable, Dict, Any, List, Tuple


def _get_defaults(to_call: Callable) -> Dict[str, Any]:
    return {param.name: param.default
            for param in signature(to_call).parameters.values()
            if param.default != param.empty}


class FactoryRegistry:
    dataloaders: Dict[str, Dict[str, Any]] = {}
    model_architectures: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}

    @classmethod
    def register_dataloader(cls, name: str):
        def wrap(dataloader: Callable):
            if name in cls.dataloaders:
                print(f"Warning: {name} already registered. Replacing with {dataloader.__name__}")
            cls.dataloaders[name] = _get_defaults(dataloader)
            return dataloader
        return wrap

    @classmethod
    def register_model_architecture(cls, name: str, compatibilities: List[str]):
        def wrap(model_arch_creator: Callable):
            if name in cls.model_architectures:
                print(f"Warning: {name} already registered. Replacing with {model_arch_creator.__name__}")
            cls.model_architectures[name] = (_get_defaults(model_arch_creator), compatibilities)

            return model_arch_creator
        return wrap
