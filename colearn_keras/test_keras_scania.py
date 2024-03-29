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
import json
import time
from colearn_grpc.example_mli_factory import ExampleMliFactory
from colearn_grpc.grpc_server import GRPCServer
from colearn_grpc.logging import get_logger
from colearn_grpc.example_grpc_learner_client import ExampleGRPCLearnerClient

# Register scania models and dataloaders in the FactoryRegistry
# pylint: disable=W0611
import colearn_keras.keras_scania  # type:ignore # noqa: F401


_logger = get_logger(__name__)


def test_keras_scania_with_grpc_sever():
    _logger.info("setting up the grpc server ...")

    server_port = 34567
    server_key = ""
    server_crt = ""
    enable_encryption = False

    server = GRPCServer(
        mli_factory=ExampleMliFactory(),
        port=server_port,
        enable_encryption=enable_encryption,
        server_key=server_key,
        server_crt=server_crt,
    )

    server.run(wait_for_termination=False)

    time.sleep(2)

    client = ExampleGRPCLearnerClient(
        "scania_client", f"127.0.0.1:{server_port}", enable_encryption=enable_encryption
    )

    client.start()

    ml = client.get_supported_system()
    data_loader = "KERAS_SCANIA"
    model_architecture = "KERAS_SCANIA"
    assert data_loader in ml["data_loaders"].keys()
    assert model_architecture in ml["model_architectures"].keys()

    data_location = "gs://colearn-public/scania/0"
    assert client.setup_ml(
        data_loader,
        json.dumps({"location": data_location}),
        model_architecture,
        json.dumps({})
    )

    weights = client.mli_propose_weights()
    assert weights.weights is not None

    client.mli_accept_weights(weights)
    assert client.mli_get_current_weights().weights == weights.weights

    client.stop()
    server.stop()
