#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import argparse
import json
import os
from typing import Optional, Sequence

from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_grpc.example_mli_factory import ExampleMliFactory
from colearn_other.demo_utils import get_split_to_folders, get_score_name

# These are imported so that they are registered in the FactoryRegistry
# pylint: disable=W0611
import colearn_keras.keras_mnist  # type:ignore # noqa: F401
import colearn_keras.keras_cifar10  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_xray  # type:ignore # noqa: F401
import colearn_pytorch.pytorch_covid_xray  # type:ignore # noqa: F401
import colearn_other.fraud_dataset  # type:ignore # noqa: F401

"""
Collective learning demo:

Demo for running all available colearn_examples
"""

mli_fac = ExampleMliFactory()
model_names = list(mli_fac.get_models().keys())

parser = argparse.ArgumentParser(description='Run colearn demo')
parser.add_argument("-d", "--data_dir", default=None,
                    help="Directory containing train data, not required for MNIST and CIFAR10")
parser.add_argument("-e", "--test_dir", default=None,
                    help="Optional directory containing test data. "
                         "Fraction of training set will be used as test set when not specified")

parser.add_argument("-m", "--model", default=model_names[0],
                    help="Model to train, options are " + " ".join(model_names))

parser.add_argument("-n", "--n_learners", default=5, type=int, help="Number of learners")
parser.add_argument("-p", "--n_rounds", default=15, type=int, help="Number of training rounds")

parser.add_argument("-v", "--vote_threshold", default=0.5, type=float,
                    help="Minimum fraction of positive votes to accept new model")

parser.add_argument("-r", "--train_ratio", default=None, type=float,
                    help="Fraction of training dataset to be used as test set when no test set is specified")

parser.add_argument("-s", "--seed", type=int, default=None,
                    help="Seed for initialising model and shuffling datasets")
parser.add_argument("-l", "--learning_rate", type=float, default=None, help="Learning rate for optimiser")
parser.add_argument("-b", "--batch_size", type=int, default=None, help="Size of training batch")

args = parser.parse_args()

model_name = args.model
dataloader_set = mli_fac.get_compatibilities()[model_name]
dataloader_name = next(iter(dataloader_set))  # use the first dataloader

n_learners = args.n_learners
test_data_folder = args.test_dir
train_data_folder = args.data_dir
vote_threshold = args.vote_threshold
n_rounds = args.n_rounds

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


score_name = get_score_name(model_name)  # get score_name for accuracy plot

split_to_folders = get_split_to_folders(dataloader_name)  # get function to split data

# split training data
train_data_folders = split_to_folders(
    data_dir=train_data_folder or "",
    n_learners=n_learners,
    train=True,
    **learning_kwargs)

# split test data
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
    model_default_params = mli_fac.get_models()[model_name]
    for key in model_default_params.keys():
        if key in learning_kwargs:
            model_default_params[key] = learning_kwargs[key]

    dataloader_default_params = mli_fac.get_dataloaders()[dataloader_name]
    for key in dataloader_default_params.keys():
        if key in learning_kwargs:
            dataloader_default_params[key] = learning_kwargs[key]
    dataloader_default_params["location"] = train_data_folders[i]
    if "test_location" in dataloader_default_params:
        dataloader_default_params["test_location"] = test_data_folders[i]

    model = mli_fac.get_mli(model_name=model_name, model_params=json.dumps(model_default_params),
                            dataloader_name=dataloader_name, dataset_params=json.dumps(dataloader_default_params))

    all_learner_models.append(model)

# Ensure all learners have same weights
set_equal_weights(all_learner_models)

# Now we're ready to start collective learning
# Get initial accuracy
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name=score_name)

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )
    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()
