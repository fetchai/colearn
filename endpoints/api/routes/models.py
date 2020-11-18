import json
from typing import Optional, Union

from fastapi import APIRouter

from api.database import DBModel
from api.schemas import ModelList, TrainedModel, Model, CopyParams
from api.utils import paginate

router = APIRouter()


def _dbmodel_to_model(rec: DBModel) -> Union[TrainedModel, Model]:
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


@router.get('/models/', response_model=ModelList, tags=['models'])
def get_list_of_models(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get a list of the models that are present on the system

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might response with fewer.
    """
    model_list = [_dbmodel_to_model(rec) for rec in DBModel.select()]

    return paginate(ModelList, model_list, page, page_size)


@router.post('/models/', tags=['models'])
def create_new_model(model: Union[TrainedModel, Model]):
    """
    Create a new model
    """
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
    return {}


@router.get('/models/{name}/', tags=['models'])
def get_specific_model_information(name: str):
    """
    Lookup model information about a specific model

    Route Parameters:

    * `name` - The name of the model to be queried
    """
    rec = DBModel.get(DBModel.name == name)
    rec.weights = None
    return _dbmodel_to_model(rec)


@router.post('/models/{name}/', tags=['models'])
def update_specific_model_information(name: str, model: Union[TrainedModel, Model]):
    """
    Update a specific model

    Route Parameters:

    * `name` - The name of the model to be updated
    """
    DBModel.update({DBModel.model: model.model,
                    DBModel.parameters: json.dumps(model.parameters),
                    DBModel.weights: json.dumps(model.weights) if hasattr(model, "weights") else None
                    }
                   ).where(DBModel.name == name).execute()
    return {}


@router.delete('/models/{name}/', tags=['models'])
def delete_specific_model(name: str):
    """
    Delete a specific model

    Route Parameters:

    * `name` - The name of the model to be deleted
    """
    rec = DBModel.get(DBModel.name == name)
    rec.delete_instance()

    return {}


@router.get('/models/{name}/export/', response_model=TrainedModel, tags=['models'])
def export_model(name: str):
    """
    Export a trained model with the parameters and trained weights. This can be then consumed by a default machine
    learning pipeline

    Route Parameters:

    * `name` - The name of the model to be exported
    """
    rec = DBModel.get(DBModel.name == name)
    return _dbmodel_to_model(rec)


@router.post('/models/{name}/copy/', tags=['models'])
def duplicate_model(name: str, params: CopyParams):
    """
    Create a copy of the specified model

    """
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
    return {}
