import logging
import requests
import utils.simulation_utils as sim
import utils.preprocessing as prep
from json import dumps as json_dumps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from logging.config import dictConfig
from log_config import log_config


dictConfig(log_config)
logger = logging.getLogger("Trainer")

# TODO config file
DATA_URL = 'http://data:5000/'

app = FastAPI()


@app.get("/health")
async def health_root():
    logger.info("Trainer: Health request received.")
    return "Trainer is online."


@app.post('/listen')
async def listen(first:  str = Form(...), second: str = Form(...)):
    print(f'logger> Trainer: Received message from {second}')
    reply = f'{round(float(first)*10,0)}'
    return {"Number": reply}


@app.get("/train")
async def train(initial_step: int):
    data = prep.get_data(initial_step, data_url=DATA_URL)
    data = prep.trainable_data(data)
    rmse_candidate, rmse_prod = prep.train_models(*data)
    return json_dumps({'rmse_candidate': rmse_candidate, 'rmse_prod': rmse_prod})


