import numpy as np
import torch.nn as nn
import torch.nn.functional as nn_func
import tensorflow as tf

from ..keras_learner import KerasLearner
from ..pytorch_learner import PytorchLearner


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

        return model


class MNISTPytorchLearner(PytorchLearner):
    def _get_model(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = nn_func.relu(self.conv1(x))
                x = nn_func.max_pool2d(x, 2, 2)
                x = nn_func.relu(self.conv2(x))
                x = nn_func.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 50)
                x = nn_func.relu(self.fc1(x))
                x = self.fc2(x)
                return nn_func.log_softmax(x, dim=1)

        model = Net()
        return model
