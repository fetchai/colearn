import numpy as np

from tensorflow.compat import v1 as tf

from keras_learner import KerasLearner


tf.disable_v2_behavior()


class MNISTConvLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 1), name="Input"
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
            self.config.n_classes, activation="softmax", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model


class MNISTSuperminiLearner(KerasLearner):
    def _get_model(self):
        def dw_block(new_layer, prev_l):
            x_dw = new_layer(prev_l)
            x_dw = tf.keras.layers.BatchNormalization()(x_dw)
            x_dw = tf.keras.layers.Dropout(rate=0.1)(x_dw)
            return x_dw

        input_img = tf.keras.Input(
            (self.config.width, self.config.height, 1), dtype=np.float32
        )

        x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu")(input_img)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)

        x = tf.keras.layers.SeparableConv2D(
            26, (3, 3), depth_multiplier=1, activation="relu"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)

        sh_l = tf.keras.layers.SeparableConv2D(
            26, (3, 3), depth_multiplier=1, padding="same", activation="relu"
        )

        for _ in range(3):
            x = dw_block(sh_l, x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(self.config.n_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        # compile model & add optimiser
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(loss=self.config.loss, metrics=["accuracy"], optimizer=opt)
        return model
