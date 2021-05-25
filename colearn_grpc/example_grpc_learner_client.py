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
import time
import traceback
import ssl

import grpc
from google.protobuf import empty_pb2

import colearn_grpc.proto.generated.interface_pb2 as ipb2
import colearn_grpc.proto.generated.interface_pb2_grpc as ipb2_grpc
from colearn.ml_interface import MachineLearningInterface, ProposedWeights, Weights
from colearn_grpc.logging import get_logger
from colearn_grpc.utils import iterator_to_weights, weights_to_iterator

_logger = get_logger(__name__)


class GRPCClientException(Exception):
    pass


class ExampleGRPCLearnerClient(MachineLearningInterface):
    """
        This is the client half of the ML gRPC connection.
        This exposes methods that:
            * interrogate the ML side about which model architectures are supported
            * create the selected model
            * implement the MachineLearningInterface: all calls to the interface will go through the
              gRPC and will be executed on the ML side.
    """

    def __init__(self, name: str, address: str, enable_encryption: bool = False):
        self.name = name
        self.address = address
        self.channel = None
        self.stub: ipb2_grpc.GRPCLearnerStub
        self.enable_encryption = enable_encryption

    def start(self):
        retries = 100
        caught_exception = None
        for i in range(0, retries):
            try:
                _logger.info(f"Attempt number {i} to connect to {self.address}")

                if self.enable_encryption:
                    credentials = None

                    # Attempt to get the certificate from the server and use it to encrypt the
                    # connection. If the certificate cannot be found, try to create an unencrypted connection.
                    try:
                        assert (':' in self.address), f"Poorly formatted address, needs :port - {self.address}"
                        _logger.info(f"Connecting to server: {self.address}")
                        addr, port = self.address.split(':')
                        trusted_certs = ssl.get_server_certificate((addr, int(port)))

                        # create credentials
                        credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs.encode())
                    except ssl.SSLError as e:
                        _logger.warning(f"Encountered ssl error when attempting to get certificate from learner server: {e}")
                    except OSError:
                        _logger.warning(f"Encountered os error when attempting to get certificate from learner server: {e}")

                    if credentials:
                        _logger.info("Creating secure channel")
                        self.channel = grpc.secure_channel(self.address, credentials)
                    else:
                        _logger.warning("Creating insecure channel")
                        self.channel = grpc.insecure_channel(self.address)
                else:
                    _logger.info("Creating channel")
                    self.channel = grpc.insecure_channel(self.address)

                self.stub = ipb2_grpc.GRPCLearnerStub(self.channel)

                # Make sure query works
                self.get_supported_system()
                _logger.info(f"Successfully connected to {self.address}!")
                return
            except grpc.RpcError as e:
                _logger.warning("gRPC error when trying to connect:")
                _logger.warning(traceback.format_exc(limit=1, chain=False))
                caught_exception = e
                time.sleep(5)
            except Exception as e:   # pylint: disable=W0703
                _logger.warning("Non grpc-based exception when trying to connect:")
                _logger.warning(traceback.format_exc(limit=1, chain=False))
                caught_exception = e
                time.sleep(5)

        _logger.exception(f"Failed to connect! Quitting... {caught_exception}")
        raise GRPCClientException("Failed to connect")

    def stop(self):
        self.channel.close()

    def get_supported_system(self):
        """ Get the supported systems from the learner, and return in the format: """
        request = empty_pb2.Empty()
        response = self.stub.QuerySupportedSystem(request)
        r = {
            "data_loaders": {},
            "model_architectures": {},
            "compatibilities": {}
        }
        for d in response.data_loaders:
            r["data_loaders"][d.name] = d.default_parameters
        for m in response.model_architectures:
            r["model_architectures"][m.name] = m.default_parameters
        for c in response.compatibilities:
            r["compatibilities"][c.model_architecture] = c.dataloaders
        return r

    def setup_ml(self, dataset_loader_name, dataset_loader_parameters,
                 model_arch_name, model_parameters):

        _logger.info(f"Setting up ml: model_arch: {model_arch_name}, dataset_loader: {dataset_loader_name}")
        _logger.debug(f"Model params: {model_parameters}")
        _logger.debug(f"Dataloader params: {dataset_loader_parameters}")

        request = ipb2.RequestMLSetup()
        request.dataset_loader_name = dataset_loader_name
        request.dataset_loader_parameters = dataset_loader_parameters
        request.model_arch_name = model_arch_name
        request.model_parameters = model_parameters

        _logger.info(f"Setting up ml with request: {request}")

        try:
            status = self.stub.MLSetup(request)
        except grpc.RpcError as ex:
            _logger.exception(f"Failed to setup MLModel: {ex}")
            return False

        return status.status == ipb2.MLSetupStatus.SUCCESS

    def mli_propose_weights(self) -> Weights:
        request = empty_pb2.Empty()
        try:
            response = self.stub.ProposeWeights(request)
            weights = iterator_to_weights(response, decode=False)
            return weights

        except grpc.RpcError as ex:
            _logger.exception(f"Failed to train_model: {ex}")
            raise ConnectionError(f"GRPC error: {ex}")

    def mli_test_weights(self, weights: Weights = None) -> ProposedWeights:
        try:
            if weights:
                response = self.stub.TestWeights(weights_to_iterator(weights, encode=False))
            else:
                raise Exception("mli_test_weights(None) is not currently supported")

            return ProposedWeights(
                weights=weights,
                vote_score=response.vote_score,
                test_score=response.test_score,
                vote=response.vote
            )
        except grpc.RpcError as ex:
            _logger.exception(f"Failed to test_model: {ex}")
            raise ConnectionError(f"GRPC error: {ex}")

    def mli_accept_weights(self, weights: Weights):
        try:
            request_iterator = weights_to_iterator(weights, encode=False)
            self.stub.SetWeights(request_iterator)
        except grpc.RpcError as e:
            _logger.exception(f"Failed to call SetWeights: {e}")
            raise ConnectionError(f"GRPC error: {e}")

    def mli_get_current_weights(self) -> Weights:
        request = empty_pb2.Empty()
        try:
            response = self.stub.GetCurrentWeights(request)
            weights = iterator_to_weights(response, decode=False)
            return weights

        except grpc.RpcError as ex:
            _logger.exception(f"Failed to get_current_weights: {ex}")
            raise Exception(f"Failed to get_current_weights: {ex}")
