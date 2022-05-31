from numpy.lib.stride_tricks import as_strided
from numpy import convolve, ones, array
from typing import Tuple


def rolling_window(x, window):
    shape = (x.size - window + 1, window)
    strides = (x.itemsize, x.itemsize)
    return as_strided(x, shape=shape, strides=strides)


def moving_average(x, steps):
    return convolve(x, ones(steps), 'valid') / steps


def seq2inputs(sequence: array, time_step_to_predict: int = 1) -> Tuple:
    X = sequence[:-1,:]
    y = sequence[:,time_step_to_predict][1:]
    print(f'Shape of feature: {X.shape} - Shape of target: {y.shape}')
    return X, y


def split_dataset(X: array, y: array, split_size: float=0.8, verbose: bool=False) -> Tuple:
    n_train_total = int(len(X) * split_size)
    n_train = int(n_train_total * split_size)
    if verbose:
        print(f'n_train: {n_train} - n_train_pre: {n_train_total}')

    X_train = X[:n_train]
    X_val = X[n_train:n_train_total]
    X_test = X[n_train_total: ]
    y_train = y[:n_train]
    y_val = y[n_train:n_train_total]
    y_test = y[n_train_total: ]
    return X_train, X_val, X_test, y_train, y_val, y_test