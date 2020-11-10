import numpy as np

import tensorflow as tf

from ..keras_learner import KerasLearner


class CovidXrayLearner(KerasLearner):
    def _get_model(self):
        inp = tf.keras.layers.Input((self.config.feature_size,))
        x = tf.keras.layers.Dense(128, activation='sigmoid')(inp)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(self.config.n_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        opt = self.config.optimizer(
            lr=self.config.l_rate, decay=self.config.l_rate_decay
        )

        model.compile(optimizer=opt, loss=self.config.loss,metrics=self.config.metrics)
        return model
