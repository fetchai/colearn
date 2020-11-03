import os
import pickle
import tempfile
from pathlib import Path


import numpy as np

from colearn_examples.config import ModelConfig
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
from colearn.basic_learner import LearnerData

import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from keras.utils import to_categorical

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
                     output_folder=Path(tempfile.gettempdir()) / "covid_xray"):
    # Load data
    covid_features=sio.loadmat(os.path.join(data_dir,'covid.mat')) 
    covid_features=covid_features['covid']

    normal_features=sio.loadmat(os.path.join(data_dir,'normal.mat')) 
    normal_features=normal_features['normal']

    pneumonia_features=sio.loadmat(os.path.join(data_dir,'pneumonia.mat')) 
    pneumonia_features=pneumonia_features['pneumonia']

    X=np.concatenate((covid_features[:,:-1],normal_features[:,:-1],pneumonia_features[:,:-1]), axis=0)#inputs
    y=np.concatenate((covid_features[:,-1],normal_features[:,-1],pneumonia_features[:,-1]), axis=0)#target labels

    min_max_scaler=MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    
    transformer = KernelPCA(n_components=64, kernel='linear')
    X = transformer.fit_transform(X)
    y = to_categorical(y)
    print("SHAPE X: ", X.shape)
    print("SHAPE Y: ", y.shape)
    
    [X, y] = shuffle_data(
        [X, y], seed=shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [X, y], data_split
    )

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.makedirs(str(dir_name), exist_ok=True)

        pickle.dump(all_images_lists[i], open(dir_name / IMAGE_FL, "wb"))
        pickle.dump(all_labels_lists[i], open(dir_name / LABEL_FL, "wb"))

        dir_names.append(dir_name)

    print(dir_names)
    return [str(x) for x in dir_names]


def prepare_single_client(config: ModelConfig, data_dir):
    data = LearnerData()
    data.train_batch_size = config.batch_size

    images = pickle.load(open(Path(data_dir) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_dir) / LABEL_FL, "rb"))

    [[train_images, test_images], [train_labels, test_labels]] = \
        split_by_chunksizes([images, labels], [config.train_ratio, config.test_ratio])

    data.train_data_size = len(train_images)

    data.train_gen = train_generator(
        train_images, train_labels, config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment
    )
    data.val_gen = train_generator(
        train_images, train_labels, config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment
    )

    data.test_data_size = len(test_images)

    data.test_gen = train_generator(
        test_images,
        test_labels,
        config.batch_size,
        config.width,
        config.height,
        config.generator_seed,
        config.train_augment,
    )

    data.test_batch_size = config.batch_size
    return data


def train_generator(data, labels, batch_size, width, height, seed,
                    augmentation=True, shuffle=True):
    # Get total number of samples in the data
    n_data = len(data)

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros(
        (batch_size, data[0].shape[0]), dtype=np.float32
    )
    batch_labels = np.zeros((batch_size, labels[0].shape[0]), dtype=np.uint8)

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
