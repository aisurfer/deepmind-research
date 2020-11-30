#!/bin/bash

# Build Dockerfile

NAME_CONTAINER="test"
PATH_DIR="/home/tony/deepmind-research"
docker build -t $NAME_CONTEINER .

# RUN CONTAINER

docker run --runtime=nvidia --rm -it -v $PATH_DIR:/tmp/deepmind-research -w /tmp/deepmind $NAME_CONTAINER bash

