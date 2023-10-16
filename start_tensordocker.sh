#!/bin/bash
# Usage: ./start_docker.sh  0 to start docker with current file-dir, anything you modify in docker will be saved in current file-dir
# Usage: ./start_docker.sh  1 to start docker only for image transportation.
# Please do not move this file to other dir, it will cause the docker container can not find the current dir.

DOCKERIMAGE="d2slam:jetson_orin_base_35.3.1"
xhost +
echo "[INFO] Start docker container with mapping current dir to docker container"
CURRENT_DIR=$(pwd)
echo "${CURRENT_DIR} will be mapped in to docker container with start option 1"
docker run -it --rm --runtime=nvidia --gpus all  --net=host -v ${CURRENT_DIR}:/root/workspace/ \
  -v /dev/:/dev/  --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name="tensor_container" ${DOCKERIMAGE} /bin/bash 