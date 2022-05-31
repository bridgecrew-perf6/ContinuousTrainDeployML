from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import logging
from logging.config import dictConfig
from log_config import log_config


dictConfig(log_config)
logger = logging.getLogger("Trainer")

app = FastAPI()


@app.get("/health")
def health_root():
    logger.info("Trainer: Health request received.")
    return "Trainer is online."

@app.get("")
def health_root():
    logger.info("Trainer: Health request received.")
    return "Trainer is up and running."

@app.post('/listen')
def _file_upload(
        # my_file: UploadFile = File(...),
        first:  str = Form(...),
        second: str = Form(...),
):
    print(f'logger> Trainer: Received message from {second}')
    reply = f'{round(float(first)*10,0)}'

    return {
        # "name": my_file.filename,
        "Number": reply,
    }