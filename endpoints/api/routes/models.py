from typing import Optional, Union

from fastapi import APIRouter

from api.schemas import ModelList, TrainedModel, Model, CopyParams

router = APIRouter()


@router.get('/models/', response_model=ModelList, tags=['models'])
def get_list_of_models(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get a list of the models that are present on the system

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    return {}


@router.post('/models/', tags=['models'])
def create_new_model(model: Union[TrainedModel, Model]):
    """
    Create a new model
    """
    return {}


@router.get('/models/{name}/', tags=['models'])
def get_specific_model_information(name: str):
    """
    Lookup model information about a specific model

    Route Parameters:

    * `name` - The name of the model to be queried
    """
    return {}


@router.post('/models/{name}/', response_model=Model, tags=['models'])
def update_specific_model_information(name: str, model: Union[TrainedModel, Model]):
    """
    Update a specific model

    Route Parameters:

    * `name` - The name of the model to be updated
    """
    return {}


@router.delete('/models/{name}/', tags=['models'])
def delete_specific_model(name: str):
    """
    Delete a specific model

    Route Parameters:

    * `name` - The name of the model to be deleted
    """
    return {}


@router.post('/models/{name}/export/', response_model=TrainedModel, tags=['models'])
def export_model(name: str):
    """
    Export a trained model with the parameters and trained weights. This can be then consumed by a default machine
    learning pipeline

    Route Parameters:

    * `name` - The name of the model to be exported
    """
    return {}


@router.post('/models/{name}/copy/', tags=['models'])
def duplicate_model(name: str, params: CopyParams):
    """
    Create a copy of the specified model

    """
    return {}
