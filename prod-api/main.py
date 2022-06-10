import logging
import requests
import pickle
import os
import utils.metadata as meta
import utils.preprocessing as prep
from prometheus_client import Gauge, start_http_server
from numpy import array
from pathlib import Path
from shutil import copyfileobj
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from logging.config import dictConfig
from log_config import log_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model


dictConfig(log_config)
logger = logging.getLogger("production")

# TODO config file
DATA_URL = 'http://data:5000/'
METRIC_PORT = 9091
bucket = 'project-capstone-fbf'
model_path = 'models/production.h5'


app = FastAPI()
meta.download_blob(bucket, model_path, model_path)
# TODO delete listdir -> to logging
print(os.listdir('models'))
track_predictions = Gauge('ProductionPredictions', 'Actual model predictions')
track_actual_data = Gauge('ActualData', 'Actual data used for predictions')


@app.get("/health")
def health():
    logger.info("Prod: Health request received.")
    return "Production Server is online."


@app.get("/predict")
async def model_prediction(initial_step: int, n_timesteps: int):
    data = prep.get_data(initial_step, data_url=DATA_URL, n_timesteps=n_timesteps)
    X, y = prep.trainable_data(data, split=False)
    prod_model = load_model('models/production.h5')
    preds = prod_model.predict(X, verbose=0)
    return make_prediction_response(preds, y)


@track_predictions.track_inprogress()
@track_actual_data.track_inprogress()
def make_prediction_response(predictions: array, actual: array):
    predictions = predictions[:,0].reshape(-1)
    actual = actual.tolist()
    predictions = predictions.tolist()
    for p, a in zip(predictions, actual):
        track_predictions.set(p)    
        track_actual_data.set(a)
    return JSONResponse({'prod_predictions': predictions})


@app.post("/evaluate")
async def evaluate_model(data: UploadFile):    
    filepath = Path('data_arrays.pkl')
    with filepath.open('wb') as buffer:
        copyfileobj(data.file, buffer)

    with open(filepath, 'rb') as f:
        data_arrays = pickle.load(f)
        
    prod_model = load_model('models/production.h5')
    return JSONResponse({'rmse_prod': round(prod_model.evaluate(*data_arrays, verbose=0)[0],2)})


@app.on_event('startup')
def startup_events():
  # start prometheus server / metrics
  start_http_server(port=METRIC_PORT)


@app.get("/trainer_connection_health")
async def trainer_server_health():
    """ 
        checks connection with trainer sending data and expecting 
        the result of a simple fixed operation
    """
    response = requests.post('http://trainer:5000/listen',data={'first': "9.99", 'second': "Prod"})
    return response.json()


@app.get("/data_connection_health")
async def data_server_health():
    """ 
        checks connection with data sending data and expecting 
        the result of a simple fixed operation
    """
    return JSONResponse(requests.post(DATA_URL+'listen', data={'first': "9.99", 'second': "Prod"}).json())
