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

from colearn.training import set_equal_weights, initial_result, collective_learning_round
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_grpc.example_grpc_learner_client import ExampleGRPCLearnerClient

cli_args = argparse.ArgumentParser(description='Start multiple GRPC learner servers')
cli_args.add_argument('-p', '--port', type=int, default=9995, help='first server port')
cli_args.add_argument('-n', '--n_learners', type=int, default=5, help='number of learners')
cli_args.add_argument('-d', '--dataloader_tag', type=str, help='dataloader tag')
cli_args.add_argument('-m', '--model_tag', type=str, help='number of learners')
cli_args.add_argument('-l', '--data_locations', type=str,
                      help='A comma-separated list of folders where the data is located. If the list has only one '
                           'item then all the learners will use the same location.')
cli_args.add_argument('-r', "--n_rounds", default=15, type=int, help="Number of training rounds")

args = cli_args.parse_args()

data_folders = args.data_locations.split(",")
if len(data_folders) == 1:
    data_folders = data_folders * args.n_learners
elif len(data_folders) != args.n_learners:
    raise Exception(f"Not enough data locations given: {data_folders}")

# Now make the corresponding grpc clients
all_learner_models = []
for i in range(args.n_learners):
    port = args.port + i
    ml_system = ExampleGRPCLearnerClient(f"client {i}", f"127.0.0.1:{port}")
    ml_system.start()
    dataloader_params = {"location": data_folders[i]}
    ml_system.setup_ml(dataset_loader_name=args.dataloader_tag,
                       dataset_loader_parameters=json.dumps(dataloader_params),
                       model_arch_name=args.model_tag,
                       model_parameters=json.dumps({}))
    all_learner_models.append(ml_system)

# now colearn as usual!
set_equal_weights(all_learner_models)

# Train the model using Collective Learning
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name="accuracy")

n_rounds = args.n_rounds
vote_threshold = 0.5
for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )

    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")

for model in all_learner_models:
    model.stop()
