import json
from typing import Optional

import peewee
from fastapi import APIRouter, HTTPException

from api.database import DBDataset
from api.schemas import Dataset, Loader, DatasetList, ErrorResponse, Empty
from api.utils import paginate_db

router = APIRouter()


def _convert_dataset(rec: DBDataset) -> Dataset:
    ds = Dataset(name=rec.name,
                 loader=Loader(name=rec.loader_name, params=json.loads(rec.loader_params)),
                 location=rec.location,
                 seed=rec.seed,
                 train_size=rec.train_size,
                 validation_size=rec.validation_size,
                 test_size=rec.test_size
                 )
    return ds


@router.get('/datasets/', response_model=DatasetList, tags=['datasets'])
def get_list_datasets(page: Optional[int] = None, page_size: Optional[int] = None):
    """
    Get the list of datasets that are present on this learner.

    Optional parameters:

    * `page` - The page index to be retrieved
    * `page_size` - The desired page size for the response. Note the server will never respond with more entries than
      specified, however, it might respond with fewer.

    """
    return paginate_db(DBDataset.select(), DatasetList, _convert_dataset, page, page_size)


@router.get(
    '/datasets/{name}/',
    response_model=Dataset,
    tags=['datasets'],
    responses={
        404: {"description": "Dataset not found", 'model': ErrorResponse},
    })
def get_specific_dataset_information(name: str):
    """
    Lookup details on a specific dataset

    Route Parameters:

    * `name` - The name of the dataset to be looked up

    """
    try:
        return _convert_dataset(DBDataset.get(DBDataset.name == name))
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Experiment not found")


@router.post(
    '/datasets/',
    response_model=Empty,
    tags=['datasets'],
    responses={
        409: {"description": "Duplicate dataset name", 'model': ErrorResponse},
    })
def create_new_dataset(dataset: Dataset):
    """
    Create / register a new dataset with the learner
    """
    try:
        DBDataset.create(name=dataset.name, loader_name=dataset.loader.name,
                         loader_params=json.dumps(dataset.loader.params),
                         location=dataset.location,
                         seed=dataset.seed,
                         train_size=dataset.train_size,
                         validation_size=dataset.validation_size,
                         test_size=dataset.test_size
                         )

    except peewee.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate dataset name")

    return {}


@router.delete(
    '/datasets/{name}/',
    response_model=Empty,
    tags=['datasets'],
    responses={
        404: {"description": "Dataset not found", 'model': ErrorResponse},
    })
def delete_dataset(name: str):
    """
    Delete the specified dataset

    Route parameters:

    * `name` - The name of the dataset to be deleted
    """
    try:
        rec = DBDataset.get(DBDataset.name == name)
        rec.delete_instance()
    except peewee.DoesNotExist:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {}
