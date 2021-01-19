import tensorflow as tf
import tensorflow_datasets as tfds

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import plot_results, plot_votes
from colearn.utils.results import Results
from colearn_keras.new_keras_learner import NewKerasLearner

"""
CIFAR10 training example using Tensorflow Keras

Used dataset:
- CIFAR10 is set of 60 000 colour images of size 32x32x3 in 10 classes

What script does:
- Loads CIFAR10 dataset from torchvision.datasets
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

n_learners = 5
n_epochs = 20
make_plot = True
vote_threshold = 0.5

width = 32
height = 32
n_classes = 10

optimizer = tf.keras.optimizers.Adam
l_rate = 0.001
batch_size = 64
loss = "sparse_categorical_crossentropy"
vote_batches = 2

train_datasets = tfds.load('cifar10',
                           split=tfds.even_splits('train', n=n_learners),
                           as_supervised=True)

test_datasets = tfds.load('cifar10',
                          split=tfds.even_splits('test', n=n_learners),
                          as_supervised=True)


def normalize_img(image, label):
    """Normalizes images: `uint8` 0-255 -> `float32` 0.0-1.0"""
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
        shape=(width, height, 3), name="Input"
    )
    x = tf.keras.layers.Conv2D(
        32, (5, 5), activation="relu", padding="same", name="Conv1_1"
    )(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(
        32, (5, 5), activation="relu", padding="same", name="Conv2_1"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Conv2D(
        64, (5, 5), activation="relu", padding="same", name="Conv3_1"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        64, activation="relu", name="fc1"
    )(x)
    x = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="fc2"
    )(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = optimizer(
        lr=l_rate
    )
    model.compile(
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(NewKerasLearner(
        model=get_model(),
        train_loader=train_datasets[i],
        test_loader=test_datasets[i],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": 100},
        model_evaluate_kwargs={"steps": vote_batches}
    ))

set_equal_weights(all_learner_models)

results = Results()
# Get initial score
results.data.append(initial_result(all_learner_models))

for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    if make_plot:
        # then make an updating graph
        plot_results(results, n_learners, block=False,
                     score_name=all_learner_models[0].criterion)
        plot_votes(results, block=False)

if make_plot:
    plot_results(results, n_learners, block=False,
                 score_name=all_learner_models[0].criterion)
    plot_votes(results, block=True)
