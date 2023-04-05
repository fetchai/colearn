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
import os
from colearn.ml_interface import PredictionRequest
from colearn_grpc.example_mli_factory import ExampleMliFactory
from colearn_grpc.grpc_server import GRPCServer
from colearn_grpc.logging import get_logger
from colearn_grpc.example_grpc_learner_client import ExampleGRPCLearnerClient

# Register mnist models and dataloaders in the FactoryRegistry
# pylint: disable=W0611
import colearn_keras.keras_mnist  # type:ignore # noqa: F401


_logger = get_logger(__name__)


def test_grpc_server_with_example_grpc_learner_client():
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
        "mnist_client", f"127.0.0.1:{server_port}", enable_encryption=enable_encryption
    )

    client.start()

    ml = client.get_supported_system()
    data_loader = "KERAS_MNIST"
    prediction_data_loader = "KERAS_MNIST_PRED"
    model_architecture = "KERAS_MNIST"
    assert data_loader in ml["data_loaders"].keys()
    assert prediction_data_loader in ml["prediction_data_loaders"].keys()
    assert model_architecture in ml["model_architectures"].keys()

    data_location = "gs://colearn-public/mnist/2/"
    assert client.setup_ml(
        data_loader,
        json.dumps({"location": data_location}),
        model_architecture,
        json.dumps({}),
        prediction_data_loader
    )

    weights = client.mli_propose_weights()
    assert weights.weights is not None

    client.mli_accept_weights(weights)
    assert client.mli_get_current_weights().weights == weights.weights

    pred_name = "prediction_1"

    rel_path = "../tests/test_data/img_0.jpg"
    location = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)

    prediction = client.mli_make_prediction(
        PredictionRequest(name=pred_name, input_data=bytes(location, 'utf-8'),
                          pred_dataloader_key="KERAS_MNIST_PRED_TWO")
    )
    prediction_data = list(prediction.prediction_data)
    assert prediction.name == pred_name
    assert isinstance(prediction_data, list)

    # Take prediction data loader from experiment
    prediction = client.mli_make_prediction(
        PredictionRequest(name=pred_name, input_data=bytes(location, 'utf-8'))
    )
    prediction_data = list(prediction.prediction_data)
    assert prediction.name == pred_name
    assert isinstance(prediction_data, list)

    client.stop()
    server.stop()
