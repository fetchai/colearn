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
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn.ml_interface import DiffPrivConfig
from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import normalize_img

n_learners = 5

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 20 if not testing_mode else 1
vote_threshold = 0.5

width = 28
height = 28
n_classes = 10

l_rate = 0.001
batch_size = 64
vote_batches = 2

# Differential privacy parameters
num_microbatches = 4  # how many batches to split a batch into
diff_priv_config = DiffPrivConfig(
    target_epsilon=1.0,  # epsilon budget for the epsilon-delta DP
    target_delta=1e-5,  # delta budget for the epsilon-delta DP
    max_grad_norm=1.5,
    noise_multiplier=1.3  # more noise -> more privacy, less utility
)


train_datasets, info = tfds.load('mnist',
                                 split=tfds.even_splits('train', n=n_learners),
                                 as_supervised=True, with_info=True)
n_datapoints = info.splits['train'].num_examples

test_dataset = tfds.load('mnist', split='test', as_supervised=True)
vote_datasets = [test_dataset.shard(num_shards=2 * n_learners, index=i) for i in range(n_learners)]
test_datasets = [test_dataset.shard(num_shards=2 * n_learners, index=i) for i in range(n_learners, 2 * n_learners)]

for i in range(n_learners):
    ds_train = train_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(n_datapoints // n_learners)
    # tf privacy expects fix batch sizes, thus drop_remainder=True
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    train_datasets[i] = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_vote = vote_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_vote = ds_vote.batch(batch_size)
    ds_vote = ds_vote.cache()
    ds_vote = ds_vote.prefetch(tf.data.experimental.AUTOTUNE)
    vote_datasets[i] = ds_vote

    ds_test = test_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    test_datasets[i] = ds_test


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

    opt = DPKerasAdamOptimizer(
        l2_norm_clip=diff_priv_config.max_grad_norm,
        noise_multiplier=diff_priv_config.noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=l_rate)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            # need to calculare the loss per sample for the
            # per sample / per microbatch gradient clipping
            reduction=tf.losses.Reduction.NONE
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model(),
        train_loader=train_datasets[i],
        vote_loader=test_datasets[i],
        test_loader=test_datasets[i],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_evaluate_kwargs={"steps": vote_batches},
        diff_priv_config=diff_priv_config
    ))

set_equal_weights(all_learner_models)

results = Results()
# Get initial score
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
