from sklearn.linear_model import SGDClassifier

import tensorflow as tf

from ..keras_learner import KerasLearner
from ..sklearn_learner import SKLearnLearner


class FraudDense1Learner(KerasLearner):
    def _get_model(self):
        model_input = tf.keras.Input(shape=self.config.input_classes, name="Input")

        x = tf.keras.layers.Dense(512, activation="relu")(model_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)

        model = tf.keras.Model(inputs=model_input, outputs=x)

        return model


class FraudSVMLearner(SKLearnLearner):
    def _get_model(self):
        return SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
