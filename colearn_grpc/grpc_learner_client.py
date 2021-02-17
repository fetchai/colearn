import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
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
        active = True
        while active:
            self._check_queue.put(0)
            time.sleep(self._health_check_time)
            with self._mu:
                active = self._active

    def _status_check_trigger(self):
        active = True
        while active:
            self._check_queue.get()
            request = ipb2.RequestStatus()
            yield request
            with self._mu:
                active = self._mu

    def _ml_system_health_loop(self):
        health_stream = self.stub.StatusStream(self._status_check_trigger())
        for pong in health_stream:
            with self._state_mu:
                self.current_state = pong.status

    @_time_query.time()
    def get_supported_system(self):
        request = empty_pb2.Empty()
        response = self.stub.QuerySupportedSystem(request)
        r = {
            "datasets": {},
            "model_architectures": {},
            "compatibility": {}
        }
        for d in response.datasets:
            r["datasets"][d.name] = d.default_parameters
        for m in response.model_architectures:
            r["model_architectures"][m.name] = m.default_parameters
        for v1, v2 in response.compatibility.items():
            r["compatibility"][v1] = v2
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
            return Weights(weights=None)

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
            return ProposedWeights(weights=None, vote_score=0, test_score=0, vote=False)

    @_time_accept.time()
    def mli_accept_weights(self, weights: Weights):
        try:
            request_iterator = weights_to_iterator(weights, encode=False)
            self.stub.SetWeights(request_iterator)
        except grpc.RpcError as e:
            _logger.exception(f"Failed to call SetWeights: {e}")
            return False
        return True

    # TODO: Implement
    def mli_get_current_weights(self) -> Weights:
        raise NotImplementedError("mli_get_current_weights not implemented")
