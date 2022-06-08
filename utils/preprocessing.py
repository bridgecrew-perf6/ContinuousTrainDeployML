from tabnanny import verbose
import requests
import pickle
from numpy.lib.stride_tricks import as_strided
from numpy import convolve, ones, array
from typing import Tuple, Union
from numpy import frombuffer, array, vstack, hstack
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam



# TODO config file
PROD_URL = 'http://prod:5000/'


def rolling_window(x, window):
    shape = (x.size - window + 1, window)
    strides = (x.itemsize, x.itemsize)
    return as_strided(x, shape=shape, strides=strides)


def moving_average(x, steps):
    return convolve(x, ones(steps), 'valid') / steps


def seq2inputs(sequence: array, time_step_to_predict: int = 1) -> Tuple:
    X = sequence[:-1,:]
    y = sequence[:,time_step_to_predict][1:]
    print(f'Train shape of features: {X.shape} - Train shape of target: {y.shape}')
    return X, y


def split_dataset(X: array, y: array, split_size: Union[int, float]=0.8, verbose: bool=False) -> Tuple:

    if isinstance(split_size, float):            
        n_train_total = int(len(X) * split_size)
        n_train = int(n_train_total * split_size)
    else:
        n_train_total = len(X) - split_size
        n_train = int(n_train_total * 0.8)  

    X_train = X[:n_train]
    X_val = X[n_train:n_train_total]
    X_test = X[n_train_total: ]
    y_train = y[:n_train]
    y_val = y[n_train:n_train_total]
    y_test = y[n_train_total: ]

    if verbose:
        print(f'n_train: {n_train_total} - evaluation: {len(y_test)}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(initial_step: int, data_url: str) -> array:
    url = data_url + 'timesteps' 
    print(f'Initial step: {initial_step}')
    initial_step_param = {'initial_step': initial_step}
    r = requests.get(url=url, params=initial_step_param)
    return frombuffer(r.content)


def trainable_data(data: array) -> Tuple:
    sequenced = rolling_window(data, 100)
    X, y = seq2inputs(sequenced)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, split_size=100, verbose=True)
    X_train = vstack((X_train, X_val))
    y_train = hstack((y_train, y_val))

    return X_train, y_train, X_test, y_test


def train_models(X_train: array, y_train: array, X_test: array, y_test: array) -> float:
    data_arrays = [X_test,  y_test]
    candidate = load_model('models/candidate.h5')
    candidate.compile(Adam(learning_rate=0.0001),loss= MeanSquaredError(), metrics=['mse'])
    candidate.fit(X_train, y_train, epochs=3, verbose=0)
    print(f'Training done', )
    candidate.save('models/candidate.h5')

    rmse_candidate = model_evaluation('candidate.h5', *data_arrays)
    
    with open('data_arrays.pkl', 'wb') as f:
        pickle.dump(data_arrays, f)

    response = requests.post(PROD_URL+'evaluate',  files={'data': open('data_arrays.pkl',"rb")})

    response = response.json()
    rmse_prod = response['rmse_prod']

    return rmse_candidate, rmse_prod


def model_evaluation(model_name: str, X_test: array, y_test: array, verbose=0) -> float:
    candidate = load_model(f'models/{model_name}')
    return round(candidate.evaluate(X_test, y_test, verbose=0)[0],2)
