import logging
import utils.preprocessing as prep
import utils.metadata as meta
from typing import Union
from time import time
from json import dumps as json_dumps
from fastapi import FastAPI, Form
from logging.config import dictConfig
from log_config import log_config

# TODO DELETE import os when ready
import os


dictConfig(log_config)
logger = logging.getLogger("Trainer")

# TODO config file
DATA_URL = 'http://data:5000/'
BUCKET = 'project-capstone-fbf'
MODEL_FOLDER = 'models/'
GCP_MODEL_PATH =  MODEL_FOLDER + 'production.h5'
LOCAL_MODEL_PATH = MODEL_FOLDER + 'candidate.h5'

app = FastAPI()
meta.download_blob(BUCKET, GCP_MODEL_PATH, LOCAL_MODEL_PATH)
print(os.listdir('models'))


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
    

@app.get("/deploy-candidate")
async def deploy(name: Union[str, None]):

    timestamp = time.strftime('%d-%m-%Y_%H%M', time.localtime())
    new_model_name = f'{GCP_MODEL_PATH}_{timestamp}'

    meta.move_blob(BUCKET, GCP_MODEL_PATH, new_model_name)

    if name is not None:
        GCP_MODEL_PATH = MODEL_FOLDER + name
        
    meta.upload_blob(BUCKET, LOCAL_MODEL_PATH, GCP_MODEL_PATH)
    return f'Deployed to {GCP_MODEL_PATH}'

