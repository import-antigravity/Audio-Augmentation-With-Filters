import os
from typing import Tuple, List, Iterable

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from tensorflow import keras

from audioaugmentation.data import window_examples, pad


def train_baseline(model, optimizer, loss, name: str, num_epochs: int, batch_size: int, window_size: int,
                   crossover: float,
                   data: Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray], callbacks: list = None,
                   model_path: str = None):
    if model_path is None:
        model_path = f'../models/{name}'
    if callbacks is None:
        callbacks = []

    callbacks += [
        keras.callbacks.ModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.4f}'),
                                        monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='auto', period=1),
        keras.callbacks.CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True)
    ]

    print(model.summary())

    X_train, y_train, X_test, y_test = data

    X_train_w, y_train_w, _ = window_examples(X_train, y_train, window_size, crossover)
    X_test_w, y_test_w, _ = window_examples(X_test, y_test, window_size, crossover)

    os.makedirs(model_path, exist_ok=True)

    print('Dataset:', X_train.shape, y_train.shape)
    model.compile(optimizer, loss=loss)
    print('Training...')
    history = model.fit(X_train_w, y_train_w, validation_data=(X_test_w, y_test_w), batch_size=batch_size,
                        epochs=num_epochs, callbacks=callbacks, shuffle=True)
    return model, optimizer, history


def train_augmented(model, optimizer, loss, name: str, num_epochs: int, batch_size: int, window_size: int,
                    crossover: float, filters: bool, salamon: bool,
                    data: Tuple[Iterable[np.ndarray], np.ndarray, Iterable[np.ndarray], np.ndarray],
                    callbacks: list = None,
                    model_path: str = None):
    if model_path is None:
        model_path = f'../models/{name}'
    if callbacks is None:
        callbacks = []

    callbacks += [
        keras.callbacks.ModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.4f}'),
                                        monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='auto', period=1),
        keras.callbacks.CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True)
    ]

    print(model.summary())

    X_train, y_train, X_test, y_test = data
    X_test_win, y_test_win, _ = window_examples(X_train, y_train, window_size, crossover)

    os.makedirs(model_path, exist_ok=True)

    print('Dataset:', X_train.shape, y_train.shape)
    model.compile(optimizer, loss=loss)

    generator = DataGenerator(X_train, y_train, filters, salamon, batch_size, window_size, crossover)

    model.compile(optimizer, loss=loss)
    print('Training...')
    history = model.fit_generator(generator, validation_data=(X_test_win, y_test_win), epochs=num_epochs,
                                  steps_per_epoch=len(generator), callbacks=callbacks, shuffle=True)
    return model, optimizer, history


class DataGenerator(object):
    def __init__(self, X, y, filters: bool, salamon: bool, batch_size: int, window_size: int, crossover: float):
        self.X = X
        self.y = y
        self.dataset_size = y.shape[0]
        self.index = np.random.permutation(self.dataset_size)
        self.batch_size = batch_size
        self.window_size = window_size
        self.crossover = crossover
        self.augment_functions = []

        if salamon:
            self.salamon_init()
        if filters:
            self.filter_df = None
            self.filter_init()
        self.state = 0

    def filter_init(self):
        df = pd.read_pickle('../data/IR16outdoor').T
        self.filter_df = df
        self.augment_functions.append(self.filter)

    def make_filters(self, n, f):
        # Choose n filters at random
        n_filters = self.filter_df.shape[1]
        filter_idx = np.random.permutation(n_filters)[:n]
        filters_df = self.filter_df[filter_idx]
        # Numpy array of shape (n, 32000)
        filters = np.nan_to_num(filters_df.to_numpy()).T
        if filters.shape[1] > 8000:
            filters = filters[:, :8000]  # Cap IR length to 32000 because we don't need IRs longer than 0.5 seconds

        # Make new filters based on laplace distribution bounded at [0, 1], mean = 0, var = 1
        coeffs = np.random.laplace(size=(f, n))  # shape (f, n)
        # bound distribution to [0, 1] and take absolute value
        coeffs = np.clip(np.abs(coeffs), 0, 1)
        # Divide such that the avg sum of coefficients is 0.5
        coeffs /= 2 * coeffs.sum(axis=1).mean()
        # create set of coefficients for delta function as 1 - sum(coeffs)
        delta_coeffs = 1 - coeffs.sum(axis=1)

        # create filters by taking outer product of coeffs and filter and summing
        delta = np.zeros(8000)
        delta[0] = 1.

        combined_filters = delta_coeffs.reshape(-1, 1) * delta
        for i in range(n):
            combined_filters += coeffs[:, i].reshape(-1, 1) * filters[i]

        return combined_filters

    def filter(self, X):
        # Choose batch_size filters at random, combine with laplace distribution
        n_filters = self.filter_df.shape[1]
        filters = self.make_filters(n_filters, self.batch_size)
        convolved = fftconvolve(X, filters, axes=1)
        return convolved

    def salamon_init(self):
        pass

    def salamon(self, X):
        return X

    def reset(self):
        self.state = 0
        self.index = np.random.permutation(self.y.shape[0])

    def __len__(self):
        return self.dataset_size

    def __next__(self):
        if self.state + self.batch_size >= self.dataset_size:
            self.reset()

        index_batch = self.index[self.state:self.state + self.batch_size]
        X_batch = pad(self.X[index_batch])
        y_batch = self.y[index_batch]
        self.state += self.batch_size

        for f in self.augment_functions:
            X_batch = f(X_batch)

        # window examples
        X_win, y_win, _ = window_examples(X_batch, y_batch, self.window_size, self.crossover)

        return X_win, y_win
