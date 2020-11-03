import numpy as np

import tensorflow as tf
import tf.keras.layers as kl
import tensorflow.keras as keras

from ..keras_learner import KerasLearner


class CovidXrayLearner(KerasLearner):
    def _get_model(self):
        inp = kl.Input((self.config.feature_size,))
        x = kl.Dense(128, activation='sigmoid')(inp)
        x = kl.Dropout(0.2)(x)
        x = kl.Dense(16, activation='sigmoid')(x)
        x = kl.Dropout(0.2)(x)
        out = kl.Dense(self.config.n_classes, activation='sigmoid')(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=self.config.optimizer, loss=self.config.loss,metrics=[keras.metrics.categorical_accuracy])
        return model
