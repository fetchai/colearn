import tensorflow as tf

from colearn_examples.config import ModelConfig

from .models import CovidXrayLearner
from .evaluation import *


class CovidXrayConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 16

        self.dataset = "420"  # or cohen

        self.metrics = ["categorical_accuracy"]

        # Model params
        self.model_type = CovidXrayLearner
        self.feature_size = 64
        self.loss = "sparse_categorical_crossentropy"
        self.n_classes = 3
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None
        self.test_ratio = 0.2

        self.val_batches = 2  # number of batches used for voting

        self.evaluation_config = {
            "auc_covid": auc_score(1),
            "auc_normal": auc_score(0),
            "auc_pneumonia": auc_score(2),
            "full_classification_report": full_classification_report(["normal", "covid", "pneumonia"], [0, 1, 2])
        }

        self.transform_metrics_for_grafana = transform_to_grafana("globaldemo_covid")
