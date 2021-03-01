# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Lock

import grpc
from google.protobuf import empty_pb2
from prometheus_client import Summary

import colearn_grpc.proto.generated.interface_pb2 as ipb2
import colearn_grpc.proto.generated.interface_pb2_grpc as ipb2_grpc
from colearn.ml_interface import MachineLearningInterface, ProposedWeights, Weights
from colearn_grpc.logging import get_logger
from colearn_grpc.utils import iterator_to_weights, weights_to_iterator

_logger = get_logger(__name__)

_time_query = Summary("contract_learner_grpc_client_query_time",
                      "Metric measures the time it takes to interrogate ML side about supported models")
_time_setup = Summary("contract_learner_grpc_client_setup_time",
                      "Metric measures the time it takes to initialize the ML side with a selected model")
_time_train = Summary("contract_learner_grpc_client_train_time",
                      "Metric measures the time it takes to train the selected model")
_time_test = Summary("contract_learner_grpc_client_test_time",
                     "Metric measures the time it takes to test")
_time_accept = Summary("contract_learner_grpc_client_accept_time",
                       "Metric measures the time it takes to accept weights")
_time_get = Summary("contract_learner_grpc_client_get_time",
                    "Metric measures the time it takes to get the current weights")


class GRPCLearnerClient(MachineLearningInterface):
    """
        This is the client half of the ML GRPC connection, which lives on the orchestrator side.
        This exposes methods:
            * to interrogate the ML side which model architectures are supported
            * to create the selected model
            * which implement MachineLearningInterface: all calls to the interface will be going trough
              GRPC and it will be executed on the ML side.

    """

    def __init__(self, name: str, address: str, health_check_time=120):
        self.name = name
        self.executor = ThreadPoolExecutor(2)
        self.address = address
        self._health_check_time = health_check_time
        self._active = True
        self._mu = Lock()
        self._check_queue: Queue = Queue()
        self.current_state = ipb2.SystemStatus.UNKNOWN
        self._state_mu = Lock()
        self.channel = None
        self.stub: ipb2_grpc.GRPCLearnerStub

    def start(self) -> bool:
        retries = 100
        ex = None
        for i in range(0, retries):
            try:
                _logger.info(f"Attempt number {i} to connect to {self.address}")
                self.channel = grpc.insecure_channel(self.address)
                self.stub = ipb2_grpc.GRPCLearnerStub(self.channel)

                self.executor.submit(self._periodic_check_trigger_generator)
                self.executor.submit(self._ml_system_health_loop)

                # Make sure query works
                self.get_supported_system()
                _logger.info(f"Successfully connected to {self.address}!")
                return True
                # TODO Update the api
            except Exception as e:  # pylint: disable=W0703
                _logger.info(f"Exception in connecting {e}")
                ex = e
                time.sleep(5)

        _logger.exception(f"Failed to connect! Quitting... {ex}")
        return False

    def stop(self):
        with self._mu:
            self._active = False
        self.executor.shutdown()
        self.channel.close()

    def current_ml_system_status(self):
        with self._state_mu:
            return self.current_state

    def trigger_ml_system_status_check(self):
        self._check_queue.put(0)

    def _periodic_check_trigger_generator(self):
        active: bool = True
        max_sleep_sec = 2
        while active:
            self._check_queue.put(0)
            # sleep is split into max_sleep_sec increments to allow for self._active check
            for _ in range(0, self._health_check_time, max_sleep_sec):
                time.sleep(max_sleep_sec)
                with self._mu:
                    active = self._active
                if not active:
                    break

    def _status_check_trigger(self):
        active: bool = True
        while active:
            try:
                # timeout is required otherwise .get() will block forever and thread will never terminate
                self._check_queue.get(block=True, timeout=2)
                request = ipb2.RequestStatus()
                yield request
            except Empty:  # raised by .get() on empty queue
                pass
            with self._mu:
                active = self._active

    def _ml_system_health_loop(self):
        health_stream = self.stub.StatusStream(self._status_check_trigger())
        for pong in health_stream:
            with self._state_mu:
                self.current_state = pong.status

    @_time_query.time()
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
        for mc in response.compatibilities:
            r["compatibilities"][mc.model_architecture] = mc.dataloaders
        return r

    @_time_setup.time()
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

    # TODO make caller aware of stream and return futures
    @_time_train.time()
    def mli_propose_weights(self) -> Weights:
        request = empty_pb2.Empty()
        try:
            response = self.stub.ProposeWeights(request)
            weights = iterator_to_weights(response, decode=False)
            return weights

        except grpc.RpcError as ex:
            _logger.exception(f"Failed to train_model: {ex}")
            raise Exception(f"Failed to train_model: {ex}")

    # TODO: check status codes
    @_time_test.time()
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
            raise Exception(f"Failed to test_model: {ex}")

    @_time_accept.time()
    def mli_accept_weights(self, weights: Weights):
        try:
            request_iterator = weights_to_iterator(weights, encode=False)
            self.stub.SetWeights(request_iterator)
        except grpc.RpcError as e:
            _logger.exception(f"Failed to call SetWeights: {e}")
            return False
        return True

    @_time_get.time()
    def mli_get_current_weights(self) -> Weights:
        request = empty_pb2.Empty()
        try:
            response = self.stub.GetCurrentWeights(request)
            weights = iterator_to_weights(response, decode=False)
            return weights

        except grpc.RpcError as ex:
            _logger.exception(f"Failed to get_current_weights: {ex}")
            raise Exception(f"Failed to get_current_weights: {ex}")
