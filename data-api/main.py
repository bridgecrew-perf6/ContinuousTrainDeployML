import utils.simulation_utils as sim
from fastapi import FastAPI, Form, Response
import logging
from logging.config import dictConfig
from log_config import log_config


dictConfig(log_config)
logger = logging.getLogger("Trainer")

app = FastAPI()
signal = sim.signals_case_generation(n_transition_steps=500)


@app.get("/health")
async def health_root():
    logger.info("Data Simalutor: Health request received.")
    return "Data Simalutor is online."


@app.get("/timesteps")
async def data(initial_step: int, n_timesteps: int):
    data = sim.timesteps_generator(signal, start_timestep=initial_step, n_timesteps=n_timesteps) 
    data_bytes = data.tobytes() 
    headers = {'Content-Disposition': 'inline; filename="timesteps"'}
    return Response(data_bytes, headers=headers, media_type='image/png') # TODO media-type 


@app.post('/listen')
async def special_health(first:  str = Form(...), second: str = Form(...),):
    print(f'logger> Trainer: Received message from {second}')
    reply = f'{round(float(first)*10/2,0)}'
    return {"Number": reply}


@app.get('/check_data')
async def special_health(step: int):
    sim.signals_case_generation(n_transition_steps=500)
    return {'data': signal[step:step+5].tolist()}


