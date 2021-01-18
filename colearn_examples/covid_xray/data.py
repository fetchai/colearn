import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler

from colearn.basic_learner import LearnerData
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
from .config import CovidXrayConfig

# this line is a fix for np.version 1.18 making a change that imgaug hasn't
# tracked yet
if float(np.version.version[2:4]) == 18:
    # pylint: disable=W0212
    np.random.bit_generator = np.random._bit_generator

IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


def split_to_folders(data_dir,
                     shuffle_seed,
                     data_split,
                     n_learners,
                     output_folder=Path(tempfile.gettempdir()) / "covid_xray",
                     test_output_folder=Path(tempfile.gettempdir()) / "covid_xray_test",
                     test_ratio=0):
    np.random.seed(shuffle_seed)

    # Load data
    covid_features = sio.loadmat(os.path.join(data_dir, 'covid.mat'))
    covid_features = covid_features['covid']

    normal_features = sio.loadmat(os.path.join(data_dir, 'normal.mat'))
    normal_features = normal_features['normal']

    pneumonia_features = sio.loadmat(os.path.join(data_dir, 'pneumonia.mat'))
    pneumonia_features = pneumonia_features['pneumonia']

    if test_ratio > 0:
        print("Global test splitting: ", test_ratio)
        test_size = int(covid_features.shape[0] * test_ratio)

        np.random.shuffle(covid_features)
        covid_test = covid_features[:test_size]
        covid_features = covid_features[test_size:]

        np.random.shuffle(normal_features)
        normal_test = normal_features[:test_size]
        normal_features = normal_features[test_size:]

        np.random.shuffle(pneumonia_features)
        pneumonia_test = pneumonia_features[:test_size]
        pneumonia_features = pneumonia_features[test_size:]

        x_test = np.concatenate((covid_test[:, :-1], normal_test[:, :-1], pneumonia_test[:, :-1]), axis=0)
        y_test = np.concatenate((covid_test[:, -1], normal_test[:, -1], pneumonia_test[:, -1]), axis=0)
        [x_test, y_test] = shuffle_data(
            [x_test, y_test], seed=shuffle_seed
        )

    x = np.concatenate((covid_features[:, :-1], normal_features[:, :-1], pneumonia_features[:, :-1]), axis=0)
    y = np.concatenate((covid_features[:, -1], normal_features[:, -1], pneumonia_features[:, -1]), axis=0)

    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    transformer = KernelPCA(n_components=64, kernel='linear')
    x = transformer.fit_transform(x)
    print("SHAPE x: ", x.shape)
    print("SHAPE Y: ", y.shape)
    if test_ratio > 0:
        x_test = min_max_scaler.transform(x_test)
        x_test = transformer.transform(x_test)
        print("SHAPE x_test: ", x_test.shape)

    [x, y] = shuffle_data(
        [x, y], seed=shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [x, y], data_split
    )

    local_output_dir = Path(output_folder)
    local_test_output_dir = Path(test_output_folder)
    print("Local output dir: ", local_output_dir)
    print("Local test output dir: ", local_test_output_dir)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        if os.path.exists(str(dir_name)):
            shutil.rmtree(str(dir_name))
        os.makedirs(str(dir_name), exist_ok=True)
        print("Shapes for learner: ", i)
        print("       input: ", len(all_images_lists[i]), "x", all_images_lists[i][0].shape)
        print("       label (idx=0): ", all_labels_lists[i][0])
        pickle.dump(all_images_lists[i], open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(all_labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    if test_ratio > 0:
        if os.path.exists(str(local_test_output_dir)):
            shutil.rmtree(str(local_test_output_dir))
        os.makedirs(str(local_test_output_dir), exist_ok=True)
        pickle.dump(x_test, open(local_test_output_dir / IMAGE_FL, "wb"))
        pickle.dump(y_test, open(local_test_output_dir / LABEL_FL, "wb"))
        dir_names.append(local_test_output_dir)
        print("Global test set created")

    print(dir_names)
    return [str(x) for x in dir_names]


def prepare_single_client(config: CovidXrayConfig, data_dir, test_data_dir=None):
    images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    if test_data_dir is not None and test_data_dir != Path(""):
        print("Covid prepare_single_client: using global test set: ", str(test_data_dir))
        test_images = pickle.load(open(Path(test_data_dir) / IMAGE_FL, "rb"))
        test_labels = pickle.load(open(Path(test_data_dir) / LABEL_FL, "rb"))
        train_images = images
        train_labels = labels
    else:
        [[train_images, test_images], [train_labels, test_labels]] = \
            split_by_chunksizes([images, labels], [config.train_ratio, config.test_ratio])

    train_data_size = len(train_images)

    print("PREPARE TRAIN: ")
    print("         0 count ", np.count_nonzero(train_labels == 0))
    print("         1 count ", np.count_nonzero(train_labels == 1))
    print("         2 count ", np.count_nonzero(train_labels == 2))
    train_gen = train_generator(
        train_images, train_labels, config.batch_size,
        config.feature_size,
        config.generator_seed,
    )
    print("PREPARE TEST: ")
    print("         0 count ", np.count_nonzero(test_labels == 0))
    print("         1 count ", np.count_nonzero(test_labels == 1))
    print("         2 count ", np.count_nonzero(test_labels == 2))
    val_gen = train_generator(
        train_images, train_labels, config.val_batch_size,
        config.feature_size,
        config.generator_seed,
    )

    test_data_size = len(test_images)

    test_gen = train_generator(
        test_images,
        test_labels,
        config.batch_size,
        config.feature_size,
        config.generator_seed,
    )

    return LearnerData(train_gen=train_gen,
                       val_gen=val_gen,
                       test_gen=test_gen,
                       train_data_size=train_data_size,
                       test_data_size=test_data_size,
                       train_batch_size=config.batch_size,
                       test_batch_size=config.batch_size)


def train_generator(data, labels, batch_size, feature_size, seed, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)
    print("COVID train generator, input shape ", len(data), "x", data[0].shape, "; label: ", labels[0].shape)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, feature_size), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, 1), dtype=np.uint8)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n_data)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)

        np.random.shuffle(indices)
    it = 0

    # Initialize a counter
    batch_counter = 0

    while True:
        batch_data[batch_counter] = data[indices[it]]
        batch_labels[batch_counter] = labels[indices[it]]

        batch_counter += 1
        it += 1

        if it >= n_data:
            it = 0

            if shuffle:
                if seed is not None:
                    np.random.seed(seed)
                np.random.shuffle(indices)

        if batch_counter == batch_size:
            yield batch_data, batch_labels
            batch_counter = 0
