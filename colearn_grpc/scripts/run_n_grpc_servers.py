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
import os
import signal
import sys
from multiprocessing.context import Process

from prometheus_client import start_http_server

from colearn_grpc.example_mli_factory import ExampleMliFactory
from colearn_grpc.grpc_server import GRPCServer
from colearn_grpc.logging import set_log_levels, get_logger

# to run tensorflow in multiple processes on the same machine, GPU must be switched off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# These are imported so that they are registered in the FactoryRegistry
# pylint: disable=W0611
# pylint: disable=C0413
import colearn_keras.keras_mnist  # type:ignore # noqa: F401  # pylint: disable=C0413
import colearn_keras.keras_cifar10  # type:ignore # noqa: F401  # pylint: disable=C0413
import colearn_pytorch.pytorch_xray  # type:ignore # noqa: F401  # pylint: disable=C0413
import colearn_pytorch.pytorch_covid_xray  # type:ignore # noqa: F401  # pylint: disable=C0413
import colearn_other.fraud_dataset  # type:ignore # noqa: F401  # pylint: disable=C0413

_logger = get_logger(__name__)


def run_grpc_server(grpc_server, metrics_port):
    # this function runs in a new process and starts the grpc server and monitoring
    if metrics_port is not None:
        start_http_server(metrics_port)

    def signal_handler(sig, frame):
        _logger.info('Received sigterm. Killing server...')
        grpc_server.stop()
        _logger.info("...done")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    server.run()


if __name__ == "__main__":
    cli_args = argparse.ArgumentParser(description='Start multiple GRPC learner servers')
    cli_args.add_argument('-p', '--port', type=int, default=9995, help='first server port')
    cli_args.add_argument('-m', '--metrics_port', type=int, default=0,
                          help='first prometheus metrics webserver port. 0 means no metrics server.')
    cli_args.add_argument('-n', '--n_learners', type=int, default=5, help='number of learners')

    args = cli_args.parse_args()

    log_levels = {"default": "INFO"}
    set_log_levels(log_levels)

    child_processes = []
    for i in range(args.n_learners):
        port = args.port + i
        if args.metrics_port != 0:
            metrics_port = args.metrics_port + i
        else:
            metrics_port = None
        server = GRPCServer(mli_factory=ExampleMliFactory(),
                            port=port)
        server_process = Process(target=run_grpc_server,
                                 kwargs={"grpc_server": server, "metrics_port": metrics_port})

        print("starting server", i)
        server_process.start()
        child_processes.append(server_process)

    def signal_handler(sig, frame):
        _logger.info('You pressed Ctrl+C! Killing child servers.')
        for child in child_processes:
            child.terminate()
        _logger.info("...done")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
