from tensorflow.compat import v1 as tf

from ..model import KerasLearner

tf.disable_v2_behavior()


class SoundConv2DLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.freq_coeficients, self.config.max_padded_len, 1),
            name="Input",
        )
        x = tf.keras.layers.Conv2D(
            32, (4, 4), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn_1_1")(x)
        x = tf.keras.layers.Conv2D(
            48, (3, 3), activation="relu", padding="same", name="Conv1_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_1_2")(x)
        x = tf.keras.layers.Conv2D(
            120, (3, 3), activation="relu", padding="same", name="Conv1_3"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_1_3")(x)

        x = tf.keras.layers.MaxPool2D(name="pool_1")(x)
        x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
        x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)

        x = tf.keras.layers.Dense(64, activation="relu", name="fc2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
        x = tf.keras.layers.Dropout(0.4, name="dropout_3")(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc3"
        )(x)

        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
