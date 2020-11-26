import tensorflow as tf

from ..keras_learner import KerasLearner


class CIFAR10Conv2Learner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 3), name="Input"
        )

        x = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv1_1",
        )(input_img)
        x = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv1_2",
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv2_1",
        )(x)
        x = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv2_2",
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv3_1",
        )(x)
        x = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            name="Conv3_2",
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(
            128, activation="relu", kernel_initializer="he_uniform", name="fc1"
        )(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc2"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        return model


class CIFAR10ConvLearner(KerasLearner):
    def _get_model(self):
        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 3), name="Input"
        )

        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_1"
        )(input_img)
        x = tf.keras.layers.BatchNormalization(name="bn1_1")(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="Conv1_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn1_2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv2_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2_1")(x)
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="Conv2_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn2_2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv3_1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3_1")(x)
        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv3_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3_2")(x)
        x = tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="Conv3_3"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn3_3")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(100, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="softmax", name="fc2"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        return model


class CIFAR10Resnet50Learner(KerasLearner):
    def _get_model(self):
        # Resnet50
        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(self.config.width, self.config.height, 3),
        )

        input_img = tf.keras.Input(
            shape=(self.config.width, self.config.height, 3), name="Input"
        )
        x = resnet(input_img)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            self.config.n_classes, activation="sigmoid", name="fc1"
        )(x)
        model = tf.keras.Model(inputs=input_img, outputs=x)

        return model
