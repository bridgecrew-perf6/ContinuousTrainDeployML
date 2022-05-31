import logging
import requests
import utils.simulation_utils as sim
import utils.preprocessing as prep
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from logging.config import dictConfig
from log_config import log_config
from numpy import frombuffer




dictConfig(log_config)
logger = logging.getLogger("Trainer")

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
    # initial_step_param = {'initial_step': 0}
    url = 'http://data:5000/timesteps'
    print(f'Initial step: {initial_step}')
    initial_step_param = {'initial_step': initial_step}
    r = requests.get(url=url, params=initial_step_param)
    data = frombuffer(r.content)
    return {'data_shape': data.shape}

