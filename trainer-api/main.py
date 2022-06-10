import logging
import utils.preprocessing as prep
import utils.metadata as meta
import time
import os 
from requests import get
from typing import Tuple
from prometheus_client import Gauge, start_http_server, Counter
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from logging.config import dictConfig
from log_config import log_config


dictConfig(log_config)
logger = logging.getLogger("Trainer")

# TODO config file
DATA_URL = 'http://data:5000/'
PROD_URL = 'http://prod:5000/'
STATIC_PROD_URL = 'http://staticprod:5000/'
BUCKET = 'project-capstone-fbf'
MODEL_FOLDER = 'models/'
MODEL_STEMMED = 'production'
MODEL_EXTENSION = '.h5'
GCP_MODEL_PATH =  MODEL_FOLDER + MODEL_STEMMED + MODEL_EXTENSION
LOCAL_MODEL_PATH = MODEL_FOLDER + 'candidate.h5'
METRIC_PORT = 9090

app = FastAPI()
meta.download_blob(BUCKET, GCP_MODEL_PATH, LOCAL_MODEL_PATH)
# TODO DELETE listdir() -> logging
print(os.listdir('models'))

track_mse_prod = Gauge('MSE_Production', 'MSE on production model evaluation')
track_mse_candidate = Gauge('MSE_Candidate', 'MSE on candidate model evaluation')
track_mse_static_prod = Gauge('MSE_Static_Production', 'MSE on candidate model evaluation')
track_predictions = Gauge('Candidate_predictions', 'Candidate Model Predictions')
track_deployments = Counter('Deployments', 'Depyloyments count')


@app.get("/health", response_class=PlainTextResponse)
async def health_root():
    logger.info("Trainer: Health request received.")
    return "Trainer is online."


@app.post('/listen')
async def listen(first:  str = Form(...), second: str = Form(...)):
    print(f'logger> Trainer: Received message from {second}')
    reply = f'{round(float(first)*10,0)}'
    return {"Number": reply}


@app.get("/train")
async def train(initial_step: int, n_timesteps: int):
    data = prep.get_data(initial_step, data_url=DATA_URL, n_timesteps=n_timesteps)
    data = prep.trainable_data(data)
    metrics = prep.train_models(*data)
    response = make_train_response(metrics)    
    return response


@track_mse_candidate.track_inprogress()
@track_mse_prod.track_inprogress()
@track_mse_static_prod.track_inprogress()
def make_train_response(metrics: Tuple):
    mse_candidate, mse_prod, mse_static_prod = metrics    
    track_mse_candidate.set(mse_candidate)
    track_mse_prod.set(mse_prod)
    track_mse_static_prod.set(mse_static_prod)

    return JSONResponse({'mse_candidate': mse_candidate, 'mse_prod': mse_prod, 'mse_static_prod': mse_static_prod})


@app.get("/predict")
def model_prediction(initial_step: int, n_timesteps: int, do_production_prediction: bool = True):
    data = prep.get_data(initial_step, data_url=DATA_URL, n_timesteps=n_timesteps)
    x_train, *_ = prep.trainable_data(data, split=False)
    preds = prep.model_prediction(x_train)
    response = make_prediction_response(preds)
    if do_production_prediction:
        params = {'initial_step': initial_step, 'n_timesteps': n_timesteps}
        _ = get(url=PROD_URL+'predict', params=params)
        _ = get(url=STATIC_PROD_URL+'predict', params=params)
    return response


@track_predictions.track_inprogress()
def make_prediction_response(predictions):
    predictions = predictions[:,0].reshape(-1)
    predictions = predictions.tolist()
    for p in predictions:
        track_predictions.set(p) 
    return JSONResponse({'train_predictions': predictions})

  
@app.get("/deploy-candidate")
async def deploy():
    timestamp = time.strftime('%d-%m-%Y_%H%M', time.localtime())
    new_model_name = f'{MODEL_FOLDER}{MODEL_STEMMED}_{timestamp}{MODEL_EXTENSION}'
    print(f'BUCKET: {BUCKET} - GCP_MODEL_PATH: {GCP_MODEL_PATH} - new_model_name: {new_model_name}')    
    meta.move_blob(BUCKET, GCP_MODEL_PATH, new_model_name)        
    meta.upload_blob(BUCKET, LOCAL_MODEL_PATH, GCP_MODEL_PATH)
    make_deploy_response()
    return PlainTextResponse(f'Deployed to {GCP_MODEL_PATH}')


@track_deployments.count_exceptions()
def make_deploy_response():
    track_deployments.inc()
    

@app.on_event('startup')
def startup_events():
  # start prometheus server / metrics
  start_http_server(port=METRIC_PORT)

