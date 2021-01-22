import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results
from colearn_keras.keras_learner import KerasLearner

n_learners = 5

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", False))  # for testing
n_epochs = 20 if not testing_mode else 1
vote_threshold = 0.5

width = 28
height = 28
n_classes = 10

l_rate = 0.001
batch_size = 64
vote_batches = 2

# Differential privacy parameters
l2_norm_clip = 1.5
noise_multiplier = 1.3  # more noise -> more privacy, less utility
num_microbatches = 64  # how many batches to split a batch into

train_datasets = tfds.load('mnist',
                           split=tfds.even_splits('train', n=n_learners),
                           as_supervised=True)

test_datasets = tfds.load('mnist',
                          split=tfds.even_splits('test', n=n_learners),
                          as_supervised=True)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


for i in range(n_learners):
    ds_train = train_datasets[i].map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(len(ds_train))
    ds_train = ds_train.batch(batch_size)
    train_datasets[i] = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

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
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=l_rate)

    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model(),
        train_loader=train_datasets[i],
        test_loader=test_datasets[i],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_evaluate_kwargs={"steps": vote_batches}
    ))

set_equal_weights(all_learner_models)

results = Results()
# Get initial score
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(n_learners=n_learners,
                   score_name=all_learner_models[0].criterion)

for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    plot.plot_results(results)
    plot.plot_votes(results)

plot.plot_results(results)
plot.plot_votes(results, block=True)

print("Colearn Example Finished!")
