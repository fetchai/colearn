import json
from typing import Optional, Union

import peewee
from fastapi import APIRouter, HTTPException

from api.database import DBModel
from api.schemas import ModelList, TrainedModel, Model, CopyParams, ErrorResponse, UpdateModel, Empty
from api.utils import paginate_db

router = APIRouter()


def _convert_model(rec: DBModel) -> Union[TrainedModel, Model]:
    if hasattr(rec, "weights") and rec.weights is not None:
        ds = TrainedModel(name=rec.name,
                          model=rec.model,
                          parameters=json.loads(rec.parameters),
                          weights=json.loads(rec.weights)
                          )
    else:
        ds = Model(name=rec.name,
                   model=rec.model,
                   parameters=json.loads(rec.parameters),
                   )
    return ds


@router.get(
    '/models/',
    response_model=ModelList,
    tags=['models']
)
def get_list_of_models(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get a list of the models that are present on the system

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might respond with fewer.
    """
    return paginate_db(DBModel.select(), ModelList, _convert_model, page, page_size)


@router.post(
    '/models/',
    response_model=Empty,
    tags=['models'],
    responses={
        409: {"description": "Duplicate model name", 'model': ErrorResponse},
    }
)
def create_new_model(model: Union[TrainedModel, Model]):
    """
    Create a new model
    """
    try:
        if hasattr(model, "weights") and model.weights is not None:
            DBModel.create(name=model.name,
                           model=model.model,
                           parameters=json.dumps(model.parameters),
                           weights=json.dumps(model.weights))
        else:
            DBModel.create(name=model.name,
                           model=model.model,
                           parameters=json.dumps(model.parameters),
                           )
    except peewee.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate model name")
    return {}


@router.get(
    '/models/{name}/',
    response_model=Model,
    tags=['models'],
    responses={
        404: {"description": "Model not found", 'model': ErrorResponse},
    })
def get_specific_model_information(name: str):
    """
    Lookup model information about a specific model

    Route Parameters:

    * `name` - The name of the model to be queried
    """
    try:
        rec = DBModel.get(DBModel.name == name)
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")

    rec.weights = None
    return _convert_model(rec)


@router.post(
    '/models/{name}/',
    tags=['models'],
    response_model=Model,
    responses={
        404: {"description": "Model not found", 'model': ErrorResponse}}
)
def update_specific_model_information(name: str, update_model: UpdateModel):
    """
    Update a specific model

    Route Parameters:

    * `name` - The name of the model to be updated
    """
    try:
        mod: DBModel = DBModel.get(DBModel.name == name)
        if update_model.weights is not None:
            mod.weights = json.dumps(update_model.weights)
        mod.save()

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")

    mod.weights = None
    return _convert_model(mod)


@router.delete(
    '/models/{name}/',
    response_model=Empty,
    tags=['models'],
    responses={
        404: {"description": "Model not found", 'model': ErrorResponse},
    })
def delete_specific_model(name: str):
    """
    Delete a specific model

    Route Parameters:

    * `name` - The name of the model to be deleted
    """
    try:
        rec = DBModel.get(DBModel.name == name)
        rec.delete_instance()

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")

    return {}


@router.get(
    '/models/{name}/export/',
    response_model=TrainedModel,
    tags=['models'],
    responses={
        404: {"description": "Model not found", 'model': ErrorResponse},
    }
)
def export_model(name: str):
    """
    Export a trained model with the parameters and trained weights. This can be then consumed by a default machine
    learning pipeline

    Route Parameters:

    * `name` - The name of the model to be exported
    """
    try:
        rec = DBModel.get(DBModel.name == name)
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")

    return _convert_model(rec)


@router.post(
    '/models/{name}/copy/',
    response_model=Empty,
    tags=['models'],
    responses={
        404: {"description": "Model not found", 'model': ErrorResponse},
        409: {"description": "Duplicate model name", 'model': ErrorResponse},
    }
)
def duplicate_model(name: str, params: CopyParams):
    """
    Create a copy of the specified model

    """
    try:
        rec = DBModel.get(DBModel.name == name)
        if params.keep_weights and hasattr(rec, "weights") and rec.weights is not None:
            DBModel.create(name=params.name,
                           model=rec.model,
                           parameters=rec.parameters,
                           weights=rec.weights)
        else:
            DBModel.create(name=params.name,
                           model=rec.model,
                           parameters=rec.parameters,
                           )
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")
    except peewee.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate model name")

    return {}
