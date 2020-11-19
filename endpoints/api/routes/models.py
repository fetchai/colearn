import json
from typing import Optional

import peewee
from fastapi import APIRouter, HTTPException

from api.database import DBModel
from api.schemas import ModelList, TrainedModel, Model, CopyParams, ErrorResponse, UpdateModel, Empty
from api.utils import paginate_db

router = APIRouter()


def _convert_trained_model(rec: DBModel) -> TrainedModel:
    return TrainedModel(
        name=rec.name,
        model=rec.model,
        parameters=json.loads(rec.parameters),
        weights=json.loads(rec.weights)
    )


def _convert_model(rec: DBModel) -> Model:
    return Model(
        name=rec.name,
        model=rec.model,
        parameters=json.loads(rec.parameters),
    )


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
def create_new_model(model: TrainedModel):
    """
    Create a new model
    """
    try:
        DBModel.create(name=model.name,
                       model=model.model,
                       parameters=json.dumps(model.parameters),
                       weights=json.dumps(model.weights) if model.weights is not None else None)
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

        return _convert_model(mod)

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")


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

    return _convert_trained_model(rec)


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

        DBModel.create(name=params.name,
                       model=rec.model,
                       parameters=rec.parameters,
                       weights=rec.weights if params.keep_weights else None)

    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model not found")
    except peewee.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate model name")

    return {}
