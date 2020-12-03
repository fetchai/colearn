#!/usr/bin/env python 
import argparse
import os

from colearn_examples.config import TrainingData, ColearnConfig
from colearn_examples.training import main

parser = argparse.ArgumentParser(description='Run colearn demo')
parser.add_argument("-d", "--data_dir", help="Directory for training data")
parser.add_argument("-t", "--task", default="XRAY",
                    help="Options are " + " ".join(str(x.name)
                                                   for x in TrainingData))
parser.add_argument("-n", "--n_learners", default=5, type=int)
parser.add_argument("-e", "--epochs", default=15, type=int)
parser.add_argument("-s", "--seed", type=int, default=None)
args = parser.parse_args()

# check data dir
if args.task == "MNIST" or args.task == "CIFAR10":  # these data are downloaded
    assert args.data_dir is None, \
        "MNIST and CIFAR10 datasets are downloaded so " \
        "data_dir should not be given"
    args.data_dir = ""
else:
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise Exception("Data dir is not a directory: " + str(data_dir))

try:
    task = TrainingData[args.task]
except KeyError:
    raise Exception("task %s not part of the TrainingData enum" % args.task)

config = ColearnConfig(task=task,
                       n_learners=args.n_learners,
                       n_epochs=args.epochs, seed=args.seed)
main(config, args.data_dir)
