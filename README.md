# ContinuousTrainDeployML
This repo contains the structure of a continuous training and deployment machine learning model. 


## Installation with conda
```
conda create -n continuous
conda install pip
pip install  -r requirements.txt
```

## Deploy Application Docker Image

```shell
sh docker-compose-build.sh
docker run --rm -it -p 5000:5000 -e MAX_WORKERS=1 key-info-extraction
```

## Start the system

```shell
docker compose up
```

