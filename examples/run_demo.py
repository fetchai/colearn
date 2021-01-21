#!/usr/bin/env python
import argparse
import os
from typing import Optional, Sequence

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import plot_results, plot_votes
from colearn.utils.results import Results
from colearn_other.mli_factory import TaskType, mli_factory

"""
Collective learning demo:

Demo for running all available examples

Arguments:
--train_dir:      Directory containing train data, not required for MNIST and CIFAR10
--test_dir:       Optional directory containing test data
                  Fraction of training set will be used as test set when not specified
--task:           Type of task for machine learning
--model_type:     Type of machine learning model, default model will be used if not specified
--n_learners:     Number of individual learners
--n_epochs:       Number of training epochs
--vote_threshold: Minimum fraction of positive votes to accept new model
--train_ratio:    Fraction of training dataset to be used as testset when no testset is specified
--seed:           Seed for initialising model and shuffling datasets
--learning_rate:  Learning rate for optimiser
--batch_size:     Size of training batch
"""

parser = argparse.ArgumentParser(description='Run colearn demo')
parser.add_argument("-d", "--train_dir", default=None, help="Directory for training data")
parser.add_argument("-e", "--test_dir", default=None, help="Directory for test data")

parser.add_argument("-t", "--task", default="KERAS_MNIST",
                    help="Options are " + " ".join(str(x.name)
                                                   for x in TaskType))

parser.add_argument("-m", "--model_type", default=None, type=str)

parser.add_argument("-n", "--n_learners", default=5, type=int, help="Number of learners")
parser.add_argument("-p", "--n_epochs", default=15, type=int, help="Number of training epochs")

parser.add_argument("-v", "--vote_threshold", default=0.5, type=float,
                    help="Minimum fraction of positive votes to accept new model")

parser.add_argument("-r", "--train_ratio", default=None, type=float,
                    help="Fraction of training dataset to be used as testset when no testset is specified")

parser.add_argument("-s", "--seed", type=int, default=None)
parser.add_argument("-l", "--learning_rate", type=float, default=None, help="Learning rate for optimiser")
parser.add_argument("-b", "--batch_size", type=int, default=None, help="Size of training batch")

args = parser.parse_args()

str_task_type = args.task
str_model_type = args.model_type
n_learners = args.n_learners
test_data_folder = args.test_dir
train_data_folder = args.train_dir
vote_threshold = args.vote_threshold
n_epochs = args.n_epochs

# Generate seed
if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), byteorder="big")

# Print seed for logs
print("Seed is ", args.seed)

# Optional arguments - will be replaced by default values depending on task/model if not set
learning_kwargs = dict()
if args.learning_rate is not None:
    learning_kwargs["learning_rate"] = args.learning_rate
if args.batch_size is not None:
    learning_kwargs["batch_size"] = args.batch_size
if args.train_ratio is not None:
    learning_kwargs["train_ratio"] = args.train_ratio

# Resolve task type
task_type = TaskType[str_task_type]

# Load correct split to folders function and resolve score_name for accuracy plot
# pylint: disable=C0415, C0412
if task_type == TaskType.PYTORCH_XRAY:
    from colearn_pytorch.pytorch_xray import split_to_folders, ModelType

    if "vote_on_accuracy" in learning_kwargs:
        if learning_kwargs["vote_on_accuracy"]:
            score_name = "auc"
        else:
            score_name = "loss"
    else:
        score_name = "auc"

elif task_type == TaskType.KERAS_MNIST:
    # noinspection PyUnresolvedReferences
    from colearn_keras.keras_mnist import split_to_folders, ModelType  # type: ignore[no-redef, misc]

    score_name = "categorical_accuracy"

elif task_type == TaskType.KERAS_CIFAR10:
    # noinspection PyUnresolvedReferences
    from colearn_keras.keras_cifar10 import split_to_folders, ModelType  # type: ignore[no-redef, misc]

    score_name = "categorical_accuracy"

elif task_type == TaskType.PYTORCH_COVID_XRAY:
    # noinspection PyUnresolvedReferences
    from colearn_pytorch.pytorch_covid_xray import split_to_folders, ModelType  # type: ignore[no-redef, misc]

    if "vote_on_accuracy" in learning_kwargs:
        if learning_kwargs["vote_on_accuracy"]:
            score_name = "categorical_accuracy"
        else:
            score_name = "loss"
    else:
        score_name = "categorical_accuracy"

elif task_type == TaskType.FRAUD:
    # noinspection PyUnresolvedReferences
    from colearn_other.fraud_dataset import split_to_folders, ModelType  # type: ignore [no-redef, misc]

    score_name = "accuracy"

else:
    raise Exception("Task %s not part of the TaskType enum" % type)

# Replace with default model type if not specified
if str_model_type is None:
    str_model_type = ModelType(1).name

# Load training data
train_data_folders = split_to_folders(
    data_dir=train_data_folder or "",
    n_learners=n_learners,
    train=True,
    **learning_kwargs)

# Load test data
test_data_folders: Sequence[Optional[str]]
if test_data_folder is not None:
    test_data_folders = split_to_folders(
        data_dir=test_data_folder,
        n_learners=n_learners,
        train=False,
        **learning_kwargs
    )
else:
    test_data_folders = [None] * n_learners

# Prepare learners
all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(mli_factory(str_task_type=str_task_type,
                                          str_model_type=str_model_type,
                                          train_folder=train_data_folders[i],
                                          test_folder=test_data_folders[i],
                                          **learning_kwargs
                                          ))

# Ensure all learners have same weights
set_equal_weights(all_learner_models)

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

for epoch in range(n_epochs):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, epoch)
    )

    plot_results(results, n_learners, score_name=score_name)
    plot_votes(results)

plot_results(results, n_learners, score_name=score_name)
plot_votes(results, block=True)
