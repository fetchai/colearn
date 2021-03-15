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
import json
import os
import pickle
from multiprocessing import Process
from pathlib import Path
from typing import Tuple

from colearn.training import collective_learning_round, set_equal_weights, initial_result
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_grpc.example_grpc_learner_client import ExampleGRPCLearnerClient
from colearn_grpc.example_mli_factory import ExampleMliFactory
from colearn_grpc.factory_registry import FactoryRegistry
from colearn_grpc.grpc_server import GRPCServer

# to run tensorflow in multiple processes on the same machine, GPU must be switched off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from colearn_keras.keras_learner import KerasLearner  # pylint: disable=C0413 # noqa: F401
from colearn_keras.keras_mnist import split_to_folders  # pylint: disable=C0413 # noqa: F401
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset  # pylint: disable=C0413 # noqa: F401
import tensorflow as tf  # pylint: disable=C0413 # noqa: F401

dataloader_tag = "KERAS_MNIST_EXAMPLE_DATALOADER"


# The dataloader needs to be registered before the models that reference it
@FactoryRegistry.register_dataloader(dataloader_tag)
def prepare_data_loaders(location: str,
                         train_ratio: float = 0.9,
                         batch_size: int = 32) -> Tuple[PrefetchDataset, PrefetchDataset]:
    """
    Load training data from folders and create train and test dataloader

    :param location: Path to training dataset
    :param train_ratio: What portion of train_data should be used as test set
    :param batch_size:
    :return: Tuple of train_loader and test_loader
    """

    data_folder = location
    image_fl = "images.pickle"
    label_fl = "labels.pickle"

    images = pickle.load(open(Path(data_folder) / image_fl, "rb"))
    labels = pickle.load(open(Path(data_folder) / label_fl, "rb"))

    n_cases = int(train_ratio * len(images))

    dataset = tf.data.Dataset.from_tensor_slices((images[:n_cases], labels[:n_cases]))
    train_loader = dataset.cache().shuffle(n_cases).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.from_tensor_slices((images[n_cases:], labels[n_cases:]))
    test_loader = dataset.cache().shuffle(len(images) - n_cases).batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    return train_loader, test_loader


model_tag = "KERAS_MNIST_EXAMPLE_MODEL"


@FactoryRegistry.register_model_architecture(model_tag, [dataloader_tag])
def prepare_learner(data_loaders: Tuple[PrefetchDataset, PrefetchDataset],
                    steps_per_epoch: int = 100,
                    vote_batches: int = 10,
                    learning_rate: float = 0.001
                    ) -> KerasLearner:
    """
    Creates new instance of KerasLearner
    :param data_loaders: Tuple of train_loader and test_loader
    :param steps_per_epoch: Number of batches per training epoch
    :param vote_batches: Number of batches to get vote_accuracy
    :param learning_rate: Learning rate for optimiser
    :return: New instance of KerasLearner
    """

    # 2D Convolutional model for image recognition
    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam

    input_img = tf.keras.Input(shape=(28, 28, 1), name="Input")
    x = tf.keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same", name="Conv1_1")(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same", name="Conv2_1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), activation="relu", padding="same", name="Conv3_1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="fc2")(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)

    opt = optimizer(lr=learning_rate)
    model.compile(loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], optimizer=opt)

    learner = KerasLearner(
        model=model,
        train_loader=data_loaders[0],
        test_loader=data_loaders[1],
        criterion="sparse_categorical_accuracy",
        minimise_criterion=False,
        model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
        model_evaluate_kwargs={"steps": vote_batches},
    )
    return learner


n_learners = 5
first_server_port = 9995
# make n servers
server_processes = []
for i in range(n_learners):
    port = first_server_port + i
    server = GRPCServer(mli_factory=ExampleMliFactory(),
                        port=port)
    server_process = Process(target=server.run)
    print("starting server", i)
    server_process.start()
    server_processes.append(server_process)

# Before we make the grpc clients, ensure that there's an mnist folder for each client
data_folders = split_to_folders(n_learners, data_split=[1 / n_learners] * n_learners)

# Now make the corresponding grpc clients
all_learner_models = []
for i in range(n_learners):
    port = first_server_port + i
    ml_system = ExampleGRPCLearnerClient(f"client {i}", f"127.0.0.1:{port}")
    ml_system.start()
    dataloader_params = {"location": data_folders[i]}
    ml_system.setup_ml(dataset_loader_name=dataloader_tag,
                       dataset_loader_parameters=json.dumps(dataloader_params),
                       model_arch_name=model_tag,
                       model_parameters=json.dumps({}))
    all_learner_models.append(ml_system)

# now colearn as usual!
set_equal_weights(all_learner_models)

# Train the model using Collective Learning
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name="accuracy")

n_rounds = 10
vote_threshold = 0.5
for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )

    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")

for model in all_learner_models:
    model.stop()

for server_process in server_processes:
    server_process.terminate()
