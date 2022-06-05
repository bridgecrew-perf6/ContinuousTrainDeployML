import utils.simulation_utils as sim
from fastapi import FastAPI, Form, Response
import logging
from logging.config import dictConfig
from log_config import log_config


dictConfig(log_config)
logger = logging.getLogger("Trainer")

signal = sim.signals_case_generation(n_transition_steps=500)

app = FastAPI()


@app.get("/health")
async def health_root():
    logger.info("Data Simalutor: Health request received.")
    return "Data Simalutor is online."


@app.get("/timesteps")
async def data(initial_step: int):
    data = sim.timesteps_generator(signal, start_timestep=initial_step, n_timesteps=4100)
    data_bytes = data.tobytes() # numpy array to bytes object
    headers = {'Content-Disposition': 'inline; filename="timesteps"'}
    return Response(data_bytes, headers=headers, media_type='image/png') # TODO media-type 


@app.post('/listen')
async def special_health(first:  str = Form(...), second: str = Form(...),):
    print(f'logger> Trainer: Received message from {second}')
    reply = f'{round(float(first)*10/2,0)}'

    return {"Number": reply}
