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
import argparse
import signal
import sys
from prometheus_client import start_http_server

from colearn_grpc.example_mli_factory import ExampleMliFactory


from colearn_grpc.grpc_server import GRPCServer
from colearn_grpc.logging import set_log_levels, get_logger


_logger = get_logger(__name__)


def create_signal_handler(server):
    def signal_handler(sig, frame):
        _logger.info('You pressed Ctrl+C! Killing server...')
        server.stop()
        _logger.info("...done")
        sys.exit(0)
    return signal_handler


def main():
    cli_args = argparse.ArgumentParser(description='Start GRPC learner server')
    cli_args.add_argument('-p', '--port', type=int, default=9995, help='server port')
    cli_args.add_argument('--metrics_port', type=int, default=9091, help='prometheus metrics webserver port')
    args = cli_args.parse_args()

    start_http_server(args.metrics_port)

    log_levels = {
        "default": "INFO"
    }

    set_log_levels(log_levels)

    server = GRPCServer(mli_factory=ExampleMliFactory(),
                        port=args.port)

    signal.signal(signal.SIGINT, create_signal_handler(server))

    server.run()


if __name__ == "__main__":
    main()
