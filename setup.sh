#!/usr/bin/env bash

cd docker

docker build -t ms-pacman-fran .

docker create -it --name=contenedor-ms-pacman-fran --runtime=nvidia ms-pacman-fran

docker start contenedor-ms-pacman-fran

docker cp contenedor-ms-pacman-fran:/estacion-de-trabajo test_folder/

docker exec -it contenedor-ms-pacman-fran ./docker-setup.sh
