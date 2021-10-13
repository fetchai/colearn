FROM python:3.7.9-slim-buster as base

USER root

WORKDIR /app

RUN apt update && \
    apt upgrade -y --no-install-recommends && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./
COPY ./setup.py ./

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY ./ ./colearn

RUN cd ./colearn && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -e .[all]

COPY colearn_grpc/scripts/run_grpc_server.py ./

COPY scripts/entrypoint.sh ./

EXPOSE 9995
EXPOSE 9091
ENV PYTHONUNBUFFERED 0
