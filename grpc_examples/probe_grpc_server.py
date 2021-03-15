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
from pprint import pprint

from colearn_grpc.example_grpc_learner_client import ExampleGRPCLearnerClient
from colearn_grpc.logging import set_log_levels

cli_args = argparse.ArgumentParser(description='Probe a GRPC learner server')
cli_args.add_argument('-p', '--port', type=int, default=9995, help='server port')
cli_args.add_argument('-i', '--ip_addr', type=str, default="127.0.0.1", help='IP address or hostname')


args = cli_args.parse_args()

# Now make a grpc client
log_levels = {"default": "INFO"}
set_log_levels(log_levels)
port = args.port
ml_system = ExampleGRPCLearnerClient("probing client", f"{args.ip_addr}:{port}")
ml_system.start()

# get info about client
ml_info = ml_system.get_supported_system()
pprint(ml_info)
ml_system.stop()
