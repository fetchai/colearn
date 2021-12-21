# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from colearn.training import set_equal_weights, initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_keras.keras_learner import KerasLearner
from colearn_other.fraud_dataset import fraud_preprocessing

"""
Fraud training example using Tensorflow Keras

Used dataset:
- Fraud, download from kaggle: https://www.kaggle.com/c/ieee-fraud-detection

What script does:
- Sets up the Keras model and some configuration parameters
- Randomly splits the dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

input_classes = 431
n_classes = 1
loss = "binary_crossentropy"
optimizer = tf.keras.optimizers.Adam
l_rate = 0.0001
l_rate_decay = 1e-5
batch_size = 10000
vote_batches = 1


def get_model():
    model_input = tf.keras.Input(shape=input_classes, name="Input")

    x = tf.keras.layers.Dense(512, activation="relu")(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        n_classes, activation="sigmoid", name="fc1"
    )(x)

    model = tf.keras.Model(inputs=model_input, outputs=x)

    opt = optimizer(
        lr=l_rate, decay=l_rate_decay
    )
    model.compile(
        loss=loss,
        metrics=[tf.keras.metrics.BinaryAccuracy()],
        optimizer=opt)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Path to data directory", type=str)
parser.add_argument("--use_cache", help="Use cached preprocessed data", type=bool, default=True)

args = parser.parse_args()

if not Path.is_dir(Path(args.data_dir)):
    sys.exit(f"Data path provided: {args.data_dir} is not a valid path or not a directory")

data_dir = args.data_dir
train_fraction = 0.9
vote_fraction = 0.05
n_learners = 5

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 7 if not testing_mode else 1

vote_threshold = 0.5
steps_per_epoch = 1

fraud_data, labels = fraud_preprocessing(data_dir, use_cache=args.use_cache)

n_datapoints = fraud_data.shape[0]
random_indices = np.random.permutation(np.arange(n_datapoints))
n_train = int(n_datapoints * train_fraction)
n_vote = int(n_datapoints * vote_fraction)
train_data = fraud_data[random_indices[:n_train]]
train_labels = labels[random_indices[:n_train]]
vote_data = fraud_data[random_indices[n_train: n_train + n_vote]]
vote_labels = labels[random_indices[n_train:n_train + n_vote]]
test_data = fraud_data[random_indices[n_train + n_vote:]]
test_labels = labels[random_indices[n_train + n_vote:]]

# make a tensorflow dataloader out of np arrays
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
vote_dataset = tf.data.Dataset.from_tensor_slices((vote_data, vote_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

# shard the dataset into n_learners pieces and add batching
train_datasets = [train_dataset.shard(num_shards=n_learners, index=i).batch(batch_size) for i in range(n_learners)]
vote_datasets = [vote_dataset.shard(num_shards=n_learners, index=i).batch(batch_size) for i in range(n_learners)]
test_datasets = [test_dataset.shard(num_shards=n_learners, index=i).batch(batch_size) for i in range(n_learners)]

all_learner_models = []
for i in range(n_learners):
    model = get_model()
    all_learner_models.append(
        KerasLearner(
            model=model,
            train_loader=train_datasets[i],
            vote_loader=vote_datasets[i],
            test_loader=test_datasets[i],
            model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
            model_evaluate_kwargs={"steps": vote_batches},
        ))

set_equal_weights(all_learner_models)

results = Results()
# Get initial score
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name="loss")

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
    print_results(results)

    # then make an updating graph
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
