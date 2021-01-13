#!/usr/bin/env python
from colearn_examples_new.new_demo import main

# Required params to be passed via arguments
str_task_type = "PYTORCH_XRAY"
str_model_type = "CONV2D"
train_data_folder = '/home/jiri/fetch/corpora/chest_xray/train'
test_data_folder = '/home/jiri/fetch/corpora/chest_xray/test'
n_learners = 5
n_epochs = 15

# Optional params
learning_kwargs = dict(
    vote_threshold=0.5,
    train_ratio=1.0,
    seed=42,
    shuffle_sees=42,
    batch_size=8,
    learning_rate=0.001,
)

main(str_task_type=str_task_type,
     train_data_folder=train_data_folder,
     test_data_folder=test_data_folder,
     n_learners=n_learners,
     n_epochs=n_epochs,
     str_model_type=str_model_type,
     **learning_kwargs)
