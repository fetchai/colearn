import argparse
import os
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

from colearn.training import set_equal_weights, initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_keras.keras_learner import KerasLearner

"""
Xray training example using Tensorflow Keras

Used dataset:
- Xray, download from kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
  The Chest X-Ray Images (Pneumonia) dataset consists of 5856 grayscale images of various sizes
  in 2 classes (normal/pneumonia).

What the script does:
- Sets up the Keras model and some configuration parameters
- Loads Xray dataset from data_dir
- Randomly splits the train folder and the test folder between multiple learners
- Does multiple rounds of learning process and displays plot with results

To Run: required argument is data_dir: Path to root folder containing data

"""

width = 128
height = 128
channels = 1
n_classes = 1
steps_per_epoch = 10
vote_batches = 13  # number of batches used for voting
optimizer = tf.keras.optimizers.Adam
l_rate = 0.001
batch_size = 8

n_learners = 5
n_rounds = 15
vote_threshold = 0.5


def get_model():
    # Minimalistic model XraySuperminiLearner
    input_img = tf.keras.Input(
        shape=(width, height, channels), name="Input"
    )
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same", name="Conv1_1"
    )(input_img)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((4, 4), name="pool1")(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="Conv2_1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)

    x = tf.keras.layers.Dense(
        n_classes, activation="sigmoid", name="fc1"
    )(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = optimizer(
        lr=l_rate
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.AUC(name='auc')],
        optimizer=opt)
    return model


# this is modified from the version in xray/data in order to keep the directory structure
# e.g. when the data is in NORMAL and PNEU directories these will also be in each of the split dirs
def split_to_folders(
        data_dir,
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=Path(tempfile.gettempdir()) / "xray",

):
    if not os.path.isdir(data_dir):
        raise Exception("Data dir does not exist: " + str(data_dir))

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    local_output_dir = Path(output_folder)

    dir_names = []
    for i in range(n_learners):
        dir_name = local_output_dir / str(i)
        os.system(f"rm -r {dir_name}")
        dir_names.append(dir_name)

    subdirs = glob(os.path.join(data_dir, "*", ""))
    for subdir in subdirs:
        subdir_name = os.path.basename(os.path.split(subdir)[0])

        cases = list(Path(subdir).rglob("*.jp*"))

        if len(cases) == 0:
            raise Exception(f"No data found in path: {str(subdir)}")

        n_cases = len(cases)
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
        random_indices = np.random.permutation(np.arange(n_cases))
        start_ind = 0

        for i in range(n_learners):
            stop_ind = start_ind + int(data_split[i] * n_cases)

            cases_subset = [cases[j] for j in random_indices[start_ind:stop_ind]]

            dir_name = local_output_dir / str(i) / subdir_name
            os.makedirs(str(dir_name))

            # make symlinks to required files in directory
            for fl in cases_subset:
                link_name = dir_name / os.path.basename(fl)
                os.symlink(fl, link_name)

            start_ind = stop_ind

    print(dir_names)
    return dir_names


# lOAD DATA
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Path to data directory", type=str)

args = parser.parse_args()

if not Path.is_dir(Path(args.data_dir)):
    sys.exit(f"Data path provided: {args.data_dir} is not a valid path or not a directory")


full_train_data_folder = os.path.join(args.data_dir, 'train')
full_test_data_folder = os.path.join(args.data_dir, 'test')

train_data_folders = split_to_folders(
    full_train_data_folder,
    shuffle_seed=42,
    n_learners=n_learners)

test_data_folders = split_to_folders(
    full_test_data_folder,
    shuffle_seed=42,
    n_learners=n_learners,
    output_folder='/tmp/xray_test'
)

train_datasets, test_datasets = [], []


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


for i in range(n_learners):
    train_datasets.append(tf.keras.preprocessing.image_dataset_from_directory(
        train_data_folders[i],
        label_mode='binary',
        batch_size=batch_size,
        image_size=(width, height),
        color_mode='grayscale'
    ).map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    test_datasets.append(tf.keras.preprocessing.image_dataset_from_directory(
        test_data_folders[i],
        label_mode='binary',
        batch_size=batch_size,
        image_size=(width, height),
        color_mode='grayscale'
    ).map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE))
# todo: augmentation (although this seems to be turned off)

all_learner_models = []
for i in range(n_learners):
    model = get_model()
    all_learner_models.append(
        KerasLearner(
            model=model,
            train_loader=train_datasets[i],
            test_loader=test_datasets[i],
            model_fit_kwargs={"steps_per_epoch": steps_per_epoch,
                              # "class_weight": {0: 1, 1: 0.27}
                              },
            model_evaluate_kwargs={"steps": vote_batches},
            criterion="auc",
            minimise_criterion=False
        ))

set_equal_weights(all_learner_models)

results = Results()
# Get initial score
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(n_learners=n_learners,
                   score_name=all_learner_models[0].criterion)

for curr_round in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, curr_round)
    )
    print_results(results)

    # then make an updating graph
    plot.plot_results(results)
    plot.plot_votes(results)

plot.plot_results(results, n_learners)
plot.plot_votes(results, block=True)

print("Colearn Example Finished!")
