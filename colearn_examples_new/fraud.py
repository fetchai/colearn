from enum import Enum
import tempfile
from pathlib import Path
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
import os
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

import pandas as pd

DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    SVM = 1


def prepare_learner(model_type: ModelType, train_loader, test_loader=None, steps_per_epoch=100,
                    vote_batches=10, **kwargs):
    return None


def prepare_data_loader(data_folder, train=True, train_ratio=0.8, batch_size=32, **kwargs):
    data = pickle.load(open(Path(data_folder) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(data_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(data))
    assert (n_cases > 0), "There are no cases"
    if train:
        data = data[:n_cases]
        labels = labels[:n_cases]
    else:
        data = data[n_cases:]
        labels = labels[n_cases:]

    return data, labels


def split_to_folders(
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=None,
        **kwargs):
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "cifar10"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    train_identity = pd.read_csv(str(Path(data_dir) / "train_identity.csv"))
    train_transaction = pd.read_csv(str(Path(data_dir) / "train_transaction.csv"))
    # Combine the data and work with the whole dataset
    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")

    del train_identity, train_transaction

    cat_cols = ["id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22",
                "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", "id_33",
                "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo", "ProductCD", "card4", "card6",
                "M4", "P_emaildomain", "R_emaildomain", "card1", "card2", "card3", "card5", "addr1", "addr2", "M1",
                "M2", "M3", "M5", "M6", "M7", "M8", "M9", "P_emaildomain_1", "P_emaildomain_2", "P_emaildomain_3",
                "R_emaildomain_1", "R_emaildomain_2", "R_emaildomain_3",
                ]

    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))

    x = train.sort_values("TransactionDT").drop(
        ["isFraud", "TransactionDT", "TransactionID"], axis=1
    )
    y = train.sort_values("TransactionDT")["isFraud"]

    del train

    # Cleaning infinite values to NaN
    x = x.replace([np.inf, -np.inf], np.nan)

    for column in x:
        x[column] = x[column].fillna(x[column].mean())

    data = x.to_numpy().astype(np.float32)
    labels = y.to_numpy().astype(np.float32)

    data = scale(data)

    [data, labels] = shuffle_data(
        [data, labels], seed=shuffle_seed
    )

    [data, labels] = split_by_chunksizes(
        [data, labels], data_split
    )

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(data[i], open(dir_name / DATA_FL, "wb"))
        pickle.dump(labels[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]
