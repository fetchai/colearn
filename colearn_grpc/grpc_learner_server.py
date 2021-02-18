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
import json
import pickle
from threading import Lock

import grpc
from google.protobuf import empty_pb2
from prometheus_client import Counter, Summary

from colearn_grpc.mli_factory_interface import MliFactory
import colearn_grpc.proto.generated.interface_pb2 as ipb2
import colearn_grpc.proto.generated.interface_pb2_grpc as ipb2_grpc
from colearn_grpc.logging import get_logger

_logger = get_logger(__name__)
_count_propose = Counter("contract_learner_grpc_server_propose",
                         "The number of times the learner has proposed weights")
_count_test = Counter("contract_learner_grpc_server_test",
                      "The number of times the learner has requested weight testing")
_count_set = Counter("contract_learner_grpc_server_set",
                     "The number of times the learner sent weights to update ML state")

_count_propose_err = Counter("contract_learner_grpc_server_propose_error",
                             "The number of errors happened during propose")
_count_test_err = Counter("contract_learner_grpc_server_test_error",
                          "The number of errors happened during testing")
_count_set_err = Counter("contract_learner_grpc_server_set_error",
                         "The number of errors happened during set weights")
_count_other_err = Counter("contract_learner_grpc_server_other_error",
                           "The number of errors happened which is not covered by specialized counters (e.g. query models or setup)")
_count_check_err = Counter("contract_learner_grpc_server_check_error",
                           "The number of how many times we tried to use the class while model not set")

_time_propose = Summary("contract_learner_grpc_server_propose_time",
                        "This metric measures the time it takes to propose weights")
_time_test = Summary("contract_learner_grpc_server_test_time",
                     "This metric measures the time it takes to test a given weight")
_time_set = Summary("contract_learner_grpc_server_set_time",
                    "This metric measures the time it takes to accept a weight")


def encode_weights(w):
    return pickle.dumps(w)


def decode_weights(w):
    return pickle.loads(w)


class GRPCLearnerServer(ipb2_grpc.GRPCLearnerServicer):
    """
        This class implements the GRPC interface methods. This class lives on the machine learning
        side of things. At construction requires ml_factory, used to create the ML model,
        which implements the MachineLearningInterface), this class will expose the interface methods
        over GRPC. When creating the object we need to specify the supported system, this describes
        the models which can be constructed with ml_factory.
    """

    def __init__(self, mli_factory: MliFactory):
        """
            @param mli_factory is a factory object that produces MachineLearningInterface objects
        """
        self.learner = None
        self._learner_mutex = Lock()
        self.mli_factory = mli_factory

    def QuerySupportedSystem(self, request, context):
        response = ipb2.ResponseSupportedSystem()
        try:
            model_to_index = {}
            for index, (name, params) in enumerate(self.mli_factory.get_models().items()):
                m = response.model_architectures.add()
                m.name = name
                m.default_parameters = json.dumps(params)
                model_to_index[name] = index
            dataloader_to_index = {}
            for index, (name, params) in enumerate(self.mli_factory.get_dataloaders().items()):
                d = response.data_loaders.add()
                d.name = name
                d.default_parameters = json.dumps(params)
                dataloader_to_index[name] = index

            for model_name, data_loaders in self.mli_factory.get_compatibilities().items():
                for dataloader_name in data_loaders:
                    response.compatibility[model_to_index[model_name]] = dataloader_to_index[dataloader_name]

        except Exception as ex:  # pylint: disable=W0703
            _logger.exception(f"Exception in QuerySupportedSystem: {ex} {type(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            _count_other_err.inc()

        return response

    def MLSetup(self, request, context):
        response = ipb2.ResponseMLSetup()
        self._learner_mutex.acquire()
        try:
            _logger.info(f"Got MLSetup request: {request}")
            self.learner = self.mli_factory.get_mli(
                model_name=request.model_arch_name,
                model_params=request.model_parameters,
                dataloader_name=request.dataset_loader_name,
                dataset_params=request.dataset_loader_parameters
            )
            _logger.debug("ML MODEL CREATED")
            if self.learner is not None:
                response.status = ipb2.MLSetupStatus.SUCCESS
            else:
                response.status = ipb2.MLSetupStatus.ERROR
        except Exception as ex:  # pylint: disable=W0703
            _logger.exception(f"Failed to create model: {ex} {type(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            response.status = ipb2.MLSetupStatus.ERROR
            _count_other_err.inc()
        finally:
            self._learner_mutex.release()
        _logger.debug(f"Sending MLSetup Response: {response}")
        return response

    def _check_model(self, context):
        with self._learner_mutex:
            if not self.learner:
                _logger.error("model not set")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("model not set up")
                _count_check_err.inc()
                return False
        return True

    @_time_propose.time()
    def ProposeWeights(self, request, context):
        _count_propose.inc()
        if not self._check_model(context):
            return
        self._learner_mutex.acquire()
        try:
            _logger.debug("Start training...")
            w = ipb2.Weights()
            weights = self.learner.mli_propose_weights()
            w.weights = encode_weights(weights)
            _logger.debug("Training done!")
            yield w
        except Exception as ex:  # pylint: disable=W0703
            _logger.exception(f"Exception in ProposeWeights: {ex} {type(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            _count_propose_err.inc()
        finally:
            self._learner_mutex.release()

    @_time_test.time()
    def TestWeights(self, request, context):
        _count_test.inc()
        pw = ipb2.ProposedWeights()
        if not self._check_model(context):
            return pw
        self._learner_mutex.acquire()
        try:
            _logger.debug("Test weights...")
            weights = decode_weights(request.weights)
            proposed_weights = self.learner.mli_test_weights(weights)
            pw.weights.weights = request.weights
            pw.vote_score = proposed_weights.vote_score
            pw.test_score = proposed_weights.test_score
            pw.vote = proposed_weights.vote
            _logger.debug("Testing done!")
        except Exception as ex:  # pylint: disable=W0703
            _logger.exception(f"Exception in TestWeights: {ex} {type(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            _count_test_err.inc()
        finally:
            self._learner_mutex.release()
        return pw

    @_time_set.time()
    def SetWeights(self, request, context):
        _count_set.inc()
        if not self._check_model(context):
            return empty_pb2.Empty()
        self._learner_mutex.acquire()
        try:
            weights = decode_weights(request.weights)
            self.learner.mli_accept_weights(weights)
        except Exception as ex:  # pylint: disable=W0703
            _logger.exception(f"Exception in SetWeights: {ex} {type(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            _count_set_err.inc()
        finally:
            self._learner_mutex.release()
        return empty_pb2.Empty()

    def StatusStream(self, request_iterator, context):
        for _ in request_iterator:
            r = ipb2.ResponseStatus()
            if self.learner:
                r.status = ipb2.SystemStatus.WORKING
            else:
                r.status = ipb2.SystemStatus.NO_MODEL
            yield r
