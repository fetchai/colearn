#!/usr/bin/env python
from torchsummary import summary

from colearn_pytorch.pytorch_xray import split_to_folders, TorchXrayModel, prepare_learner, prepare_data_loader
from colearn_examples.training import initial_result, collective_learning_round, set_equal_weights
from colearn_examples.utils.results import Results
from colearn_examples.utils.plot import plot_results, plot_votes

n_learners = 5
n_epochs = 15
vote_threshold = 0.5

learning_kwargs = dict(
    seed=42,
    shuffle_sees=42,
    batch_size=8,
    learning_rate=0.001,
)

full_train_data_folder = '/home/jiri/fetch/corpora/chest_xray/train'
full_test_data_folder = '/home/jiri/fetch/corpora/chest_xray/test'

# lOAD DATA
train_data_folders = split_to_folders(
    full_train_data_folder,
    n_learners=n_learners,
    **learning_kwargs)

test_data_folders = split_to_folders(
    full_test_data_folder,
    n_learners=n_learners,
    output_folder='/tmp/xray_test',
    **learning_kwargs
)

learner_train_dataloaders = []
learner_test_dataloaders = []

for i in range(n_learners):
    learner_train_dataloaders.append(prepare_data_loader(train_data_folders[i], **learning_kwargs))
    learner_test_dataloaders.append(prepare_data_loader(test_data_folders[i], **learning_kwargs))

all_learner_models = []
for i in range(n_learners):
    model = TorchXrayModel()
    all_learner_models.append(prepare_learner(model,
                                              learner_train_dataloaders[i],
                                              test_loader=learner_test_dataloaders[i],
                                              **learning_kwargs))

set_equal_weights(all_learner_models)
score_name = all_learner_models[0].score_name

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
