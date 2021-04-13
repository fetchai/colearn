#!/bin/bash

if test "$#" -eq 1; then
    DOMAIN=$1
    echo "Setting up self signed certificate for domain $DOMAIN"

    openssl req -newkey rsa:2048 -nodes -keyout server.key -x509 -days 365 -out server.crt -subj "/C=GB/ST=Cambridge/L=Cambridge/O=None/OU=None Department/CN=$DOMAIN"
fi

echo "Running python3 /app/run_orchestrator.py"
python3 -u /app/run_grpc_server.py
