#!/usr/bin/env python 
import argparse
import os
from pathlib import Path

from colearn.config import TrainingData, ColearnConfig
from training import main

parser = argparse.ArgumentParser(description='Run colearn demo')
parser.add_argument("-d", "--data_dir", required=True)
parser.add_argument("-t", "--task", default="XRAY")
parser.add_argument("-s", "--seed", type=int, default=None)
args = parser.parse_args()

data_dir = os.path.abspath(args.data_dir)
if not os.path.isdir(data_dir):
    raise Exception("Data dir does not exist: " + str(data_dir))

try:
    task = TrainingData[args.task]
except KeyError:
    raise Exception("task %s not part of the TrainingData enum" % args.task)

config = ColearnConfig(Path(data_dir), task, seed=args.seed)
main(config)
