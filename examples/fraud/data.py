import os
import pickle
from pathlib import Path

from basic_learner import LearnerData

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from examples.utils.data import shuffle_data, split_by_chunksizes

from .config import FraudConfig


def fraud_preprocessing(data_dir, data_file, labels_file):
    train_identity = pd.read_csv(str(Path(data_dir) / "train_identity.csv"))
    train_transaction = pd.read_csv(str(Path(data_dir) / "train_transaction.csv"))
    # Combine the data and work with the whole dataset
    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")

    del train_identity, train_transaction

    cat_cols = [
        "id_12",
        "id_13",
        "id_14",
        "id_15",
        "id_16",
        "id_17",
        "id_18",
        "id_19",
        "id_20",
        "id_21",
        "id_22",
        "id_23",
        "id_24",
        "id_25",
        "id_26",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_32",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
        "DeviceType",
        "DeviceInfo",
        "ProductCD",
        "card4",
        "card6",
        "M4",
        "P_emaildomain",
        "R_emaildomain",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "M1",
        "M2",
        "M3",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "P_emaildomain_1",
        "P_emaildomain_2",
        "P_emaildomain_3",
        "R_emaildomain_1",
        "R_emaildomain_2",
        "R_emaildomain_3",
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

    np.save(data_file, data)
    np.save(labels_file, labels)
    return data, labels


DATA_FL = "data.pickle"
LABEL_FL = "labels.pickle"


def split_to_folders(config, data_dir, output_folder=Path(os.getcwd()) / "fraud"):
    data_file = data_dir + "/data.npy"
    labels_file = data_dir + "/labels.npy"

    if not os.path.isfile(data_file) or not os.path.isfile(labels_file):
        data, labels = fraud_preprocessing(data_dir, data_file, labels_file)
    else:
        data = np.load(data_file)
        labels = np.load(labels_file)

    [data, labels] = shuffle_data([data, labels], config.shuffle_seed)

    [data_lists, labels_lists] = split_by_chunksizes([data, labels], config.data_split)

    use_cloud = False
    local_output_dir = Path(output_folder)
    bucket = None
    remote_output_dir = None

    dir_names = []
    for i in range(config.n_learners):

        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(data_lists[i], open(dir_name / DATA_FL, "wb"))
        pickle.dump(labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        if use_cloud:
            # upload files to gcloud
            remote_dir = os.path.join(remote_output_dir, str(i))
            for fl in [DATA_FL, LABEL_FL]:
                remote_image = os.path.join(remote_dir, fl)
                file_blob = bucket.blob(str(remote_image))

                # Set 5MB chunk size otherwise upload times out
                file_blob.chunk_size = (5 * 1024 * 1024)
                file_blob.upload_from_filename(str(dir_name / fl))
            dir_names.append("gs://" + bucket.name + "/" + remote_dir)
        else:
            dir_names.append(dir_name)
    return dir_names


def prepare_single_client(config: FraudConfig, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    data = pickle.load(open(Path(data_dir) / DATA_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    [[train_data, test_data], [train_labels, test_labels]] = split_by_chunksizes(
        [data, labels], [config.train_ratio, config.test_ratio]
    )

    data.train_data_size = len(train_data)

    data.train_gen = train_generator(
        train_data, train_labels, config.batch_size, config
    )
    data.val_gen = train_generator(train_data, train_labels, config.batch_size, config)

    data.test_data_size = len(test_data)

    data.test_gen = train_generator(test_data, test_labels, config.batch_size, config)

    data.test_batch_size = config.batch_size
    return data


def train_generator(data, labels, batch_size, config, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, config.input_classes), dtype=np.float32)
    batch_labels = np.zeros((batch_size, config.n_classes), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n_data)

    if shuffle:
        if config.generator_seed is not None:
            np.random.seed(config.generator_seed)
        np.random.shuffle(indices)

    it = 0

    # Initialize a counter
    batch_counter = 0

    while True:
        batch_data[batch_counter] = data[indices[it]]
        batch_labels[batch_counter] = labels[indices[it]]

        it += 1
        batch_counter += 1

        if it >= n_data:
            it = 0

            if shuffle:
                if config.generator_seed is not None:
                    np.random.seed(config.generator_seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
