FROM python:3.7.9-slim-buster as base

USER root

WORKDIR /app

RUN apt update && \
    apt upgrade -y --no-install-recommends && \
    DEBIAN_FRONTEND="noninteractive" apt install --autoremove -y python3-opencv --no-install-recommends && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./
COPY ./setup.py ./

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY ./ ./colearn

RUN cd ./colearn && \
    pip3 install --no-cache-dir -e .[keras] && \
    pip3 install --no-cache-dir -e .[grpc] && \
    pip3 install --no-cache-dir -e .[examples]

RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

USER appuser

COPY grpc_examples/run_grpc_server.py ./

EXPOSE 9995
EXPOSE 9091
ENV PYTHONUNBUFFERED 0

ENTRYPOINT [ "python3",  "/app/run_grpc_server.py"]
