import json
from typing import Optional

from fastapi import APIRouter

from api.database import DBDataset
from api.schemas import Dataset, Loader, DatasetList
from api.utils import paginate

router = APIRouter()


def _dbdataset_to_dataset(rec: DBDataset) -> Dataset:
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
      specified, however, it might response with fewer.

    """
    dataset_list = [_dbdataset_to_dataset(rec) for rec in DBDataset.select()]

    return paginate(DatasetList, dataset_list, page, page_size)


@router.get('/datasets/{name}/', response_model=Dataset, tags=['datasets'])
def get_specific_dataset_information(name: str):
    """
    Lookup details on a specific dataset

    Route Parameters:

    * `name` - The name of the dataset to be looked up

    """
    rec = DBDataset.get(DBDataset.name == name)

    return _dbdataset_to_dataset(rec)


@router.post('/datasets/', tags=['datasets'])
def create_new_dataset(dataset: Dataset):
    """
    Create / register a new dataset with the learner
    """
    d1 = DBDataset.create(name=dataset.name, loader_name=dataset.loader.name,
                          loader_params=json.dumps(dataset.loader.params),
                          location=dataset.location,
                          seed=dataset.seed,
                          train_size=dataset.train_size,
                          validation_size=dataset.validation_size,
                          test_size=dataset.test_size
                          )
    print(d1)

    return {}


@router.delete('/datasets/{name}/', tags=['datasets'])
def delete_dataset(name: str):
    """
    Delete the specified dataset

    Route parameters:

    * `name` - The name of the dataset to be deleted
    """
    rec = DBDataset.get(DBDataset.name == name)
    rec.delete_instance()

    return {}
