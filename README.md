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
```

## Start the system

```shell
docker compose up
```

## Credentials
The system requires credentials to GCP in the project mlops-3. 
The credentials have to be located in 'src/gcp_config/mlops-3-1ccb1337a897.json'
