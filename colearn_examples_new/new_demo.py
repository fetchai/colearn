#!/usr/bin/env python
from torchsummary import summary

from colearn_pytorch.pytorch_xray import XrayDataset, split_to_folders, Net
from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.results import Results
from colearn_pytorch.new_pytorch_learner import NewPytorchLearner
from colearn_examples.utils.plot import plot_results, plot_votes

import torch
import torch.nn as nn

from colearn_examples_pytorch.utils import auc_from_logits


n_learners = 5
batch_size = 8
seed = 42
n_epochs = 15
vote_threshold = 0.5
learning_rate = 0.001

pos_weight = torch.tensor([0.27])
steps_per_epoch = 10
vote_batches = 13  # number of batches used for voting
vote_using_auc = True

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# lOAD DATA
full_train_data_folder = '/home/jiri/fetch/corpora/chest_xray/train'
full_test_data_folder = '/home/jiri/fetch/corpora/chest_xray/test'
train_data_folders = split_to_folders(
    full_train_data_folder,
    shuffle_seed=42,
    n_learners=n_learners)

test_data_folders = split_to_folders(
    full_test_data_folder,
    shuffle_seed=42,
    n_learners=n_learners,
    output_folder='/tmp/xray_test'
)

learner_train_dataloaders = []
learner_test_dataloaders = []

for i in range(n_learners):
    learner_train_dataloaders.append(torch.utils.data.DataLoader(
        XrayDataset(train_data_folders[i], train_ratio=1),
        batch_size=batch_size, shuffle=True)
    )
    learner_test_dataloaders.append(torch.utils.data.DataLoader(
        XrayDataset(test_data_folders[i], train_ratio=1),
        batch_size=batch_size, shuffle=True)
    )

if vote_using_auc:
    learner_vote_kwargs = dict(
        vote_criterion=auc_from_logits,
        minimise_criterion=False)
    score_name = "auc"
else:
    learner_vote_kwargs = {}
    score_name = "loss"

# Make n instances of NewPytorchLearner with model and torch dataloaders
all_learner_models = []
for i in range(n_learners):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learner = NewPytorchLearner(
        model=model,
        train_loader=learner_train_dataloaders[i],
        test_loader=learner_test_dataloaders[i],
        device=device,
        optimizer=opt,
        criterion=nn.BCEWithLogitsLoss(
            # pos_weight=pos_weight,
            reduction='mean'),
        num_train_batches=steps_per_epoch,
        num_test_batches=vote_batches,
        **learner_vote_kwargs
    )

    all_learner_models.append(learner)

set_equal_weights(all_learner_models)

# TODO: universal way to get input_size
# print a summary of the model architecture
summary(all_learner_models[0].model, input_size=(1, 128, 128))

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
