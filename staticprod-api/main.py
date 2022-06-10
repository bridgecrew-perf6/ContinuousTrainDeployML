import logging
import pickle
import os
import utils.metadata as meta
import utils.preprocessing as prep
from prometheus_client import Gauge, start_http_server
from numpy import array
from pathlib import Path
from shutil import copyfileobj
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from logging.config import dictConfig
from log_config import log_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model


dictConfig(log_config)
logger = logging.getLogger("static production")

# TODO config file
DATA_URL = 'http://data:5000/'
METRIC_PORT = 9090
bucket = 'project-capstone-fbf'
model_path = 'models/production.h5'


app = FastAPI()
meta.download_blob(bucket, model_path, model_path)
prod_model = load_model('models/production.h5')
print('Loaded model')

track_static_predictions = Gauge('Production_predictions_static', 'Actual model predictions')


@app.get("/health", response_class=PlainTextResponse)
def health():
    logger.info("Prod: Health request received.")
    return "Static Production Server is online."


@app.get("/predict")
async def model_prediction(initial_step: int, n_timesteps: int):
    data = prep.get_data(initial_step, data_url=DATA_URL, n_timesteps=n_timesteps)
    X, _ = prep.trainable_data(data, split=False)
    preds = prod_model.predict(X, verbose=0)
    return make_prediction_response(preds)


@track_static_predictions.track_inprogress()
def make_prediction_response(predictions: array):
    predictions = predictions[:,0].reshape(-1)
    predictions = predictions.tolist()
    for p in predictions:
        track_static_predictions.set(p)    
    return JSONResponse({'prod_static_predictions': predictions})


@app.post("/evaluate")
async def evaluate_model(data: UploadFile):    
    filepath = Path('data_arrays.pkl')
    with filepath.open('wb') as buffer:
        copyfileobj(data.file, buffer)

    with open(filepath, 'rb') as f:
        data_arrays = pickle.load(f)
        
    return JSONResponse({'mse_static_prod': round(prod_model.evaluate(*data_arrays, verbose=0)[0],2)})


@app.on_event('startup')
def startup_events():
  # start prometheus server / metrics
  start_http_server(port=METRIC_PORT)
