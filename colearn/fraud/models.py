from sklearn.linear_model import SGDClassifier

from tensorflow.compat import v1 as tf

from colearn.model import KerasLearner, SKLearnLearner

tf.disable_v2_behavior()


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

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class FraudSVMLearner(SKLearnLearner):
    def _get_model(self):
        return SGDClassifier(max_iter=1, verbose=0, loss="modified_huber")
