from fastapi import FastAPI, HTTPException
import logging
from logging.config import dictConfig
from log_config import log_config
import requests

dictConfig(log_config)
logger = logging.getLogger("production")

app = FastAPI()


@app.get("/health")
def health():
    logger.info("Prod: Health request received.")
    return "Production Server is online."


@app.get("/trainer_connection_health")
async def trainer_server_health():
    response = requests.post(
    'http://trainer:5000/listen',
    data={'first': "9.99", 'second': "Prod"}
    )
    return response.json()


@app.get("/data_connection_health")
async def data_server_health():
    response = requests.post(
    'http://data:5000/listen',
    data={'first': "9.99", 'second': "Prod"}
    )
    return response.json()




@app.get("/predict")
async def predict():
    response = requests.post(
    'http://trainer:5000/listen',
    data={'first': "9.99", 'second': "Prod"}
    )
    return response.json()




    





