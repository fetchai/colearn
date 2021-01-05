from pathlib import Path

import tensorflow as tf
import numpy as np

from colearn_examples.training import set_equal_weights, initial_result, collective_learning_round
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results
from new_keras_learner import NewKerasLearner

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


data_dir = '/home/emmasmith/Development/datasets/fraud'
DATA_FL = "data.npy"
LABEL_FL = "labels.npy"
train_fraction = 0.9
n_learners = 5
n_epochs = 7
vote_threshold = 0.5
steps_per_epoch = 1

fraud_data: np.array = np.load(Path(data_dir) / DATA_FL)
labels = np.load(Path(data_dir) / LABEL_FL)
n_datapoints = fraud_data.shape[0]
random_indices = np.random.permutation(np.arange(n_datapoints))
n_train = int(n_datapoints * train_fraction)
train_data = fraud_data[random_indices[:n_train]]
train_labels = labels[random_indices[:n_train]]
test_data = fraud_data[random_indices[n_train:]]
test_labels = labels[random_indices[n_train:]]


# make a tensorflow dataloader out of np arrays
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

# shard the dataset into n_learners pieces and add batching
train_datasets = [train_dataset.shard(num_shards=n_learners, index=i).batch(batch_size) for i in range(n_learners)]
test_datasets = [test_dataset.shard(num_shards=n_learners, index=i).batch(batch_size) for i in range(n_learners)]

all_learner_models = []
for i in range(n_learners):
    model = get_model()
    all_learner_models.append(
        NewKerasLearner(
            model=model,
            train_loader=train_datasets[i],
            test_loader=test_datasets[i],
            model_fit_kwargs={"steps_per_epoch": steps_per_epoch},
            model_evaluate_kwargs={"steps": vote_batches},
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

    # then make an updating graph
    plot_results(results, n_learners, block=False, score_name="loss")
    plot_votes(results, block=False)

plot_results(results, n_learners, block=False, score_name="loss")
plot_votes(results, block=True)