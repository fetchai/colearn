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
from concurrent import futures
import os
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

    def __init__(self, mli_factory: MliFactory, port=None, max_workers=5,
                 enable_encryption=False, server_key="", server_crt=""):
        """
            @param mli_factory is a factory object that produces MachineLearningInterface objects
            @param port is the port where the server will listen
            @param max_workers is how many worker threads will be available in the thread pool
            @param enable_encryption if True they the server will try to enable encryption
            @param server_key path to server key for encryption (if enabled)
            @param server_crt path to server certificate for encryption (if enabled)
        """
        self.port = port
        self.server = None
        self.service = GRPCLearnerServer(mli_factory)
        self.thread_pool = None
        self.max_workers = max_workers
        self.enable_encryption = enable_encryption
        self.server_key = server_key
        self.server_crt = server_crt

    def run(self):
        if self.server:
            raise ValueError("re-running grpc")

        address = "0.0.0.0:{}".format(self.port)

        _logger.info(f"Starting GRPC server on {address}...")

        encrypted_connection = False
        if self.enable_encryption:
            # There needs to be a certificate and private key available to enable encryption -
            # if these are not available, fall back to no encryption.
            encrypted_connection = True

            if not os.path.isfile(self.server_crt):
                _logger.error(f"Failed to find file {self.server_crt} needed for encrypted grpc connection - not enabling")
                encrypted_connection = False

            if not os.path.isfile(self.server_key):
                _logger.error(f"Failed to find file {self.server_key} needed for encrypted grpc connection - not enabling")
                encrypted_connection = False

        self.thread_pool = futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="GRPCLearnerServer-poolworker-")

        self.server = grpc.server(self.thread_pool)
        ipb2_grpc.add_GRPCLearnerServicer_to_server(self.service, self.server)

        if encrypted_connection:
            # read in key and certificate
            with open(self.server_key, 'rb') as f:
                private_key = f.read()
            with open(self.server_crt, 'rb') as f:
                certificate_chain = f.read()

            # create server credentials
            server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain,),))
            self.server.add_secure_port(address, server_credentials)
        else:
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
