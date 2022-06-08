import logging
from typing import Tuple
import utils.preprocessing as prep
import utils.metadata as meta
import time
from prometheus_client import Gauge, start_http_server
from json import dumps as json_dumps
from fastapi import FastAPI, Form, Request
from logging.config import dictConfig
from log_config import log_config
from json import loads as json_loads

# TODO DELETE import os when ready
import os


dictConfig(log_config)
logger = logging.getLogger("Trainer")

# TODO config file
DATA_URL = 'http://data:5000/'
BUCKET = 'project-capstone-fbf'
MODEL_FOLDER = 'models/'
MODEL_STEMMED = 'production'
MODEL_EXTENSION = '.h5'
GCP_MODEL_PATH =  MODEL_FOLDER + MODEL_STEMMED + MODEL_EXTENSION
LOCAL_MODEL_PATH = MODEL_FOLDER + 'candidate.h5'
METRIC_PORT = 9090

app = FastAPI()
meta.download_blob(BUCKET, GCP_MODEL_PATH, LOCAL_MODEL_PATH)
print(os.listdir('models'))

track_mse_prod = Gauge('mse_prod', 'MSE on production model evaluation')
track_mse_candidate = Gauge('mse_candidate', 'MSE on candidate model evaluation')

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
    metrics = prep.train_models(*data)
    response = make_response(metrics)    
    return response


@track_mse_candidate.track_inprogress()
@track_mse_prod.track_inprogress()
def make_response(metrics: Tuple):

    mse_candidate, mse_prod = metrics    
    track_mse_candidate.set(mse_candidate)
    track_mse_prod.set(mse_prod)

    return json_dumps({'rmse_candidate': mse_candidate, 'rmse_prod': mse_prod})
    

@app.get("/deploy-candidate")
async def deploy():

    timestamp = time.strftime('%d-%m-%Y_%H%M', time.localtime())
    new_model_name = f'{MODEL_FOLDER}{MODEL_STEMMED}_{timestamp}{MODEL_EXTENSION}'
    print(f'BUCKET: {BUCKET} - GCP_MODEL_PATH: {GCP_MODEL_PATH} - new_model_name: {new_model_name}')    
    meta.move_blob(BUCKET, GCP_MODEL_PATH, new_model_name)        
    meta.upload_blob(BUCKET, LOCAL_MODEL_PATH, GCP_MODEL_PATH)

    return f'Deployed to {GCP_MODEL_PATH}'


@app.on_event('startup')
def startup_events():
  # start prometheus server / metrics
  start_http_server(port=METRIC_PORT)

