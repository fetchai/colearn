from tensorflow.compat import v1 as tf

from colearn.model import KerasLearner

tf.disable_v2_behavior()


class MedicalSoundConv2DLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.freq_coeficients, self.config.max_padded_len, 1),
            name="Input",
        )

        x = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn_1_1")(x)
        x = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", name="Conv1_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_1_2")(x)
        x = tf.keras.layers.MaxPool2D(name="pool_1")(x)

        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_2_1")(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv2_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_2_2")(x)
        x = tf.keras.layers.MaxPool2D(name="pool_2")(x)

        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv3_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_3_1")(x)
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv3_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_3_2")(x)
        x = tf.keras.layers.MaxPool2D(name="pool_3")(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_4")(x)

        x = tf.keras.layers.Dense(64, activation="relu", name="fc2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_5")(x)

        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc4"
        )(x)

        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
