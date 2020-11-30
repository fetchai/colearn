import tensorflow as tf

from colearn_examples.config import ModelConfig

from .models import CovidXrayLearner
from .evaluation import auc_score, full_classification_report, transform_to_grafana, get_confusion_matrix


class CovidXrayConfig(ModelConfig):
    def __init__(self, seed=None):
        super().__init__(seed)

        # Training params
        self.optimizer = tf.keras.optimizers.Adam
        self.l_rate = 0.001
        self.l_rate_decay = 1e-5
        self.batch_size = 4

        self.dataset = "420"  # or cohen

        self.metrics = ["sparse_categorical_accuracy"]

        # Model params
        self.model_type = CovidXrayLearner
        self.feature_size = 64
        self.loss = "sparse_categorical_crossentropy"
        self.n_classes = 3
        self.class_labels = [0, 1, 2]
        self.multi_hot = False

        # Data params
        self.steps_per_epoch = None
        self.test_ratio = 0.25
        self.use_dp = False

        self.val_batches = 10  # number of batches used for voting
        self.val_batch_size = 8

        self.evaluation_config = {
            #"auc_covid": auc_score(1),
            #"auc_normal": auc_score(0),
            #"auc_pneumonia": auc_score(2),
            "full_classification_report": full_classification_report(["normal", "covid", "pneumonia"], [0, 1, 2]),
            "confusion_matrix": get_confusion_matrix([0, 1, 2])
        }

        self.transform_metrics_for_grafana = transform_to_grafana("globaldemo_covid")
