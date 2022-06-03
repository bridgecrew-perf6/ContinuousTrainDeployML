import requests
import utils.simulation_utils as sim
import utils.preprocessing as prep
from numpy import frombuffer, array, vstack, hstack
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from typing import Tuple


def train(initial_step: int):
    data = get_data(initial_step)
    data = trainable_data(data)
    rmse_candidate, rmse_prod = train_model(*data)    
    return {'RMSE_candidate': rmse_candidate, 'RMSE_prod': rmse_prod}


def get_data(initial_step: int):
    n_timesteps = 4100
    data = sim.signals_case_generation(n_transition_steps=500)
    return data[initial_step:initial_step + n_timesteps]


def get_data_api(initial_step: int) -> array:
    url = data_url + 'timesteps' 
    print(f'Initial step: {initial_step}')
    initial_step_param = {'initial_step': initial_step}
    r = requests.get(url=url, params=initial_step_param)
    return frombuffer(r.content)


def trainable_data(data: array) -> Tuple:
    sequenced = prep.rolling_window(data, 100)
    X, y = prep.seq2inputs(sequenced)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_dataset(X, y, split_size=100, verbose=True)
    X_train = vstack((X_train, X_val))
    y_train = hstack((y_train, y_val))

    return X_train, y_train, X_test, y_test


def train_model(X_train: array, y_train: array, X_test: array, y_test: array) -> float:
    candidate = load_model('src/models/model_one')
    candidate.compile(Adam(learning_rate=0.0001),loss= MeanSquaredError(), metrics=['mse'])
    candidate.fit(X_train, y_train, epochs=5)
    rmse_candidate = round(candidate.evaluate(X_test, y_test, verbose=0)[0],2)
    prod = load_model('src/models/model_one')
    rmse_prod = round(prod.evaluate(X_test, y_test, verbose=0)[0],2)

    return rmse_candidate, rmse_prod


if __name__ == "__main__":
    results = train(0)
    print(results)