from typing import Optional
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn_examples.training import initial_result, collective_learning_round
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


class NewKerasLearner(MachineLearningInterface):
    def __init__(self, model,
                 train_loader,
                 test_loader=None,
                 minimise_criterion=True):

        self.model: keras.Model = model
        self.train_loader: tf.data.Dataset = train_loader
        self.test_loader: tf.data.Dataset = test_loader
        self.minimise_criterion = minimise_criterion

        self.vote_score = self.test(self.train_loader)

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_loader)

        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_score=vote_score,
                               test_score=test_score,
                               vote=vote
                               )

    def vote(self, new_score) -> bool:
        if self.minimise_criterion:
            return new_score <= self.vote_score
        else:
            return new_score >= self.vote_score

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)

    def mli_get_current_weights(self) -> Weights:
        return Weights(weights=self.model.get_weights())

    def set_weights(self, weights: Weights):
        self.model.set_weights(weights.weights)

    def train(self):
        self.model.fit(self.train_loader, epochs=1)

    def test(self, loader: tf.data.Dataset):
        loss = self.model.evaluate(x=loader)
        return loss


if __name__ == "__main__":
    n_learners = 5
    n_epochs = 20
    make_plot = True
    vote_threshold = 0.5

    width = 28
    height = 28
    n_classes = 10

    optimizer = tf.keras.optimizers.Adam
    l_rate = 0.001
    l_rate_decay = 1e-5
    batch_size = 64
    loss = "sparse_categorical_crossentropy"

    train_datasets = tfds.load('mnist',
                               split=tfds.even_splits('train', n=n_learners),
                               as_supervised=True)

    test_datasets = tfds.load('mnist',
                              split=tfds.even_splits('test', n=n_learners),
                              as_supervised=True)


    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label


    # ds_info.splits['train'].num_examples
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

        opt = optimizer(
            lr=l_rate, decay=l_rate_decay
        )
        model.compile(loss=loss, optimizer=opt)
        return model


    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(NewKerasLearner(
            model=get_model(),
            train_loader=train_datasets[i],
            test_loader=test_datasets[i],
        ))

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
            plot_results(results, n_learners, block=False)
            plot_votes(results, block=False)

    if make_plot:
        plot_results(results, n_learners, block=False)
        plot_votes(results, block=True)
