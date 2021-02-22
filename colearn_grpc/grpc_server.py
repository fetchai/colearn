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
from concurrent import futures
import grpc

from colearn_grpc.mli_factory_interface import MliFactory

from colearn_grpc.grpc_learner_server import GRPCLearnerServer
import colearn_grpc.proto.generated.interface_pb2_grpc as ipb2_grpc

from colearn_grpc.logging import get_logger


_logger = get_logger(__name__)


class GRPCServer:
    """
        This is a wrapper class, which simplify the usage of GRPCLearnerServer.
        It requires a port, ml_factory and supported_system, out of which builds GRPCLearnerServer
        object, and creates the GRPC listener server, which can be started using the run method.
    """

    def __init__(self, mli_factory: MliFactory, port=None, max_workers=5):
        """
            @param mli_factory is a factory object that produces MachineLearningInterface objects
            @param port is the port where the server will listen
            @param max_workers is how many worker threads will be available in the thread pool
        """
        self.port = port
        self.server = None
        self.service = GRPCLearnerServer(mli_factory)
        self.thread_pool = None
        self.max_workers = max_workers

    def run(self):
        if self.server:
            raise ValueError("re-running grpc")

        address = "0.0.0.0:{}".format(self.port)

        _logger.info(f"Starting GRPC server on {address}...")

        self.thread_pool = futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="GRPCLearnerServer-poolworker-")
        self.server = grpc.server(self.thread_pool)

        ipb2_grpc.add_GRPCLearnerServicer_to_server(self.service, self.server)
        self.server.add_insecure_port(address)
        self.server.start()
        _logger.info("GRPC server started. Waiting for termination...")
        self.server.wait_for_termination()

    def stop(self):
        _logger.info("Stopping GRPC server...")
        if self.server:
            self.server.stop(2).wait()
        self.server = None

        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        self.thread_pool = None
        _logger.info("server stopped")
