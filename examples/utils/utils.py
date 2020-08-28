from typing import List
import numpy as np
import os

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# Uncomment that to disable GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def missing_values_cross_entropy_loss(y_true, y_pred):
    # We're adding a small epsilon value to prevent computing logarithm of 0 (consider y_hat == 0.0 or y_hat == 1.0).
    epsilon = tf.constant(1.0e-30, dtype=np.float32)
    # Check that there are no NaN values in predictions (neural network shouldn't output NaNs).
    y_pred = tf.debugging.assert_all_finite(y_pred, "y_pred contains NaN")
    # Temporarily replace missing values with zeroes, storing the missing values mask for later.
    y_true_not_nan_mask = tf.logical_not(tf.math.is_nan(y_true))
    y_true_nan_replaced = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true
    )
    # Cross entropy, but split into multiple lines for readability:
    # y * log(y_hat)
    positive_predictions_cross_entropy = y_true_nan_replaced * tf.math.log(
        y_pred + epsilon
    )
    # (1 - y) * log(1 - y_hat)
    negative_predictions_cross_entropy = (1.0 - y_true_nan_replaced) * tf.math.log(
        1.0 - y_pred + epsilon
    )
    # c(y, y_hat) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    cross_entropy_loss = -(
        positive_predictions_cross_entropy + negative_predictions_cross_entropy
    )
    # Use the missing values mask for replacing loss values in places in which the label was missing with zeroes.
    # (y_true_not_nan_mask is a boolean which when casted to float will take values of 0.0 or 1.0)
    cross_entropy_loss_discarded_nan_labels = cross_entropy_loss * tf.cast(
        y_true_not_nan_mask, tf.float32
    )
    mean_loss_per_row = tf.reduce_mean(cross_entropy_loss_discarded_nan_labels, axis=1)
    mean_loss = tf.reduce_mean(mean_loss_per_row)
    return mean_loss


class Result:
    def __init__(self):
        self.vote = False
        self.votes = []
        self.test_accuracies = []
        self.vote_accuracies = []
        self.block_proposer = None


class Results:
    def __init__(self):
        self.data = []  # type: List[Result]

        # Data for plots and statistics
        self.h_test_accuracies = []
        self.h_vote_accuracies = []

        self.mean_test_accuracies = []
        self.mean_vote_accuracies = []

        self.max_test_accuracies = []
        self.max_vote_accuracies = []

        self.highest_test_accuracy = 0
        self.highest_vote_accuracy = 0

        self.highest_mean_test_accuracy = 0
        self.highest_mean_vote_accuracy = 0

        self.current_mean_test_accuracy = 0
        self.current_mean_vote_accuracy = 0

        self.current_max_test_accuracy = 0
        self.current_max_vote_accuracy = 0

        self.mean_mean_test_accuracy = 0
        self.mean_mean_vote_accuracy = 0
