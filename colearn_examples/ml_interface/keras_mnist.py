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
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import normalize_img

"""
MNIST training example using Keras

Used dataset:
- MNIST is set of 60 000 black and white hand written digits images of size 28x28x1 in 10 classes

What script does:
- Loads MNIST dataset from Keras
- Sets up a Keras learner
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

n_learners = 5
vote_threshold = 0.5
vote_batches = 2

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 20 if not testing_mode else 1
width = 28
height = 28
n_classes = 10
l_rate = 0.001
batch_size = 64

# Load data for each learner
train_dataset, info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
n_datapoints = info.splits['train'].num_examples

train_datasets = [train_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]

test_dataset = tfds.load('mnist', split='test', as_supervised=True)
vote_datasets = [test_dataset.shard(num_shards=2 * n_learners, index=i) for i in range(n_learners)]
test_datasets = [test_dataset.shard(num_shards=2 * n_learners, index=i) for i in range(n_learners, 2 * n_learners)]


for i in range(n_learners):
    train_datasets[i] = train_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_datasets[i] = train_datasets[i].shuffle(n_datapoints // n_learners)
    train_datasets[i] = train_datasets[i].batch(batch_size)

    vote_datasets[i] = vote_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vote_datasets[i] = vote_datasets[i].batch(batch_size)

    test_datasets[i] = test_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_datasets[i] = test_datasets[i].batch(batch_size)


# Define model
def get_model():
    input_img = tf.keras.Input(
        shape=(width, height, 1), name="Input"
    )
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="Conv1_1"
    )(input_img)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="Conv2_1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="fc1"
    )(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = tf.keras.optimizers.Adam(lr=l_rate)
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model(),
        train_loader=train_datasets[i],
        vote_loader=vote_datasets[i],
        test_loader=test_datasets[i],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_evaluate_kwargs={"steps": vote_batches},
    ))

set_equal_weights(all_learner_models)

# Train the model using Collective Learning
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name=all_learner_models[0].criterion)

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )

    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
