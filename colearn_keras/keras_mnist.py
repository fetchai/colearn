from enum import Enum
import tempfile
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from colearn_keras.new_keras_learner import NewKerasLearner
from colearn_examples.utils.data import shuffle_data
from colearn_examples.utils.data import split_by_chunksizes
import os
import pickle
import numpy as np

IMAGE_FL = "images.pickle"
LABEL_FL = "labels.pickle"


class ModelType(Enum):
    CONV2D = 1


def prepare_model(type: ModelType, **kwargs):
    if type == ModelType.CONV2D:
        return get_keras_xray_conv2D_model(**kwargs)
    else:
        raise Exception("Model %s not part of the ModelType enum" % type)


def get_keras_xray_conv2D_model(learning_rate=0.001, **kwargs):
    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam

    input_img = tf.keras.Input(
        shape=(28, 28, 1), name="Input"
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
        10, activation="softmax", name="fc2"
    )(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = optimizer(
        lr=learning_rate
    )
    model.compile(
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=opt)
    return model


def prepare_learner(model_type: ModelType, train_loader, test_loader=None, steps_per_epoch=100,
                    vote_batches=10, **kwargs):
    learner = NewKerasLearner(
        model=prepare_model(model_type, **kwargs),
        train_loader=train_loader,
        test_loader=test_loader,
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
        model_evaluate_kwargs={"steps": vote_batches},
    )
    return learner


def prepare_data_loader(data_folder, train=True, train_ratio=0.8, batch_size=32, **kwargs):
    images = pickle.load(open(Path(data_folder) / IMAGE_FL, "rb"))
    labels = pickle.load(open(Path(data_folder) / LABEL_FL, "rb"))

    n_cases = int(train_ratio * len(images))
    assert (n_cases > 0), "There are no cases"
    if train:
        images = images[:n_cases]
        labels = labels[:n_cases]
    else:
        images = images[n_cases:]
        labels = labels[n_cases:]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.cache()
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def split_to_folders(
        n_learners,
        data_split=None,
        shuffle_seed=None,
        output_folder=None,
        **kwargs
):
    if output_folder is None:
        output_folder = Path(tempfile.gettempdir()) / "mnist"

    if data_split is None:
        data_split = [1 / n_learners] * n_learners

    # Load MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Normalization
    all_images = all_images.astype("float32") / 255.0

    # Add channel dimension: 28,28 -> 28,28,1
    all_images = np.expand_dims(all_images, axis=-1)

    [all_images, all_labels] = shuffle_data(
        [all_images, all_labels], seed=shuffle_seed
    )

    [all_images_lists, all_labels_lists] = split_by_chunksizes(
        [all_images, all_labels], data_split
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
