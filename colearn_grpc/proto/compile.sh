#!/usr/bin/env bash

rm -r ./generated/*

python3 -m grpc_tools.protoc \
        -I . \
        --python_out=./generated \
        --grpc_python_out=./generated \
        *.proto

# protoc uses implicit relative imports which are not allowed in python3. This converts implicit imports of the
# form "import .*_pb2" to explicit relative imports ("from . import")
sed -i.bak '/^import\ .*_pb2/s/^/from \. /' ./generated/*.py
