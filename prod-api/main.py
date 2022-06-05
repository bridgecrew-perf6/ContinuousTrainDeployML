from copyreg import pickle
import logging
import requests
import pickle
from pathlib import Path
from shutil import copyfileobj
import utils.data_interactions as data
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from logging.config import dictConfig
from log_config import log_config
from tensorflow.keras.models import load_model
import os


dictConfig(log_config)
logger = logging.getLogger("production")

# TODO config file
DATA_URL = 'http://data:5000/listen'
bucket = 'project-capstone-fbf'
model_path = 'models/production.h5'


app = FastAPI()
client = data.gcp_client()
data.download_blob(client, bucket, model_path, model_path)
print(os.listdir('models'))


@app.get("/health")
def health():
    logger.info("Prod: Health request received.")
    return "Production Server is online."


@app.post("/evaluate")
async def evaluate_model(data: UploadFile):
    
    filepath = Path('data_arrays.pkl')
    print(f'entered evaluate model. File type: {type(data)}')
    with filepath.open('wb') as buffer:
        copyfileobj(data.file, buffer)

    with open(filepath, 'rb') as f:
        data_arrays = pickle.load(f)
        
    print(type(data_arrays))
    prod_model = load_model('models/production.h5')
    return JSONResponse({'rmse_prod': round(prod_model.evaluate(*data_arrays, verbose=0)[0],2)})



@app.get("/trainer_connection_health")
async def trainer_server_health():
    """ 
        checks connection with trainer sending data and expecting 
        the result of a simple fixed operation
    """
    response = requests.post(
    'http://trainer:5000/listen',
    data={'first': "9.99", 'second': "Prod"}
    )
    return response.json()


@app.get("/data_connection_health")
async def data_server_health():
    """ 
        checks connection with data sending data and expecting 
        the result of a simple fixed operation
    """
    return JSONResponse(requests.post(DATA_URL, data={'first': "9.99", 'second': "Prod"}).json())


# def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
#     try:
#         with destination.open("wb") as buffer:
#             copyfileobj(upload_file.file, buffer)
#     finally:
#         upload_file.file.close()
