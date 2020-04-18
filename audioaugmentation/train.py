import os
from typing import Tuple, List, Iterable, Callable, Optional

import numpy as np
import pandas as pd
from librosa import effects
from scipy.signal import fftconvolve
from tensorflow import keras
from tensorflow.python.keras.utils import Sequence

from audioaugmentation.data import window_examples, pad


def train_baseline(model, optimizer, loss, name: str, num_epochs: int, batch_size: int, window_size: int,
                   crossover: float, data: Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray],
                   callbacks: list = None, model_path: str = None, preprocess: Optional[Tuple[Callable, Tuple]] = None):
    if model_path is None:
        model_path = f'../models/{name}'
    if callbacks is None:
        callbacks = []

    callbacks += [
        keras.callbacks.ModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.4f}'),
                                        monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='auto', period=1),
        keras.callbacks.CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
    ]

    print(model.summary())

    X_train, y_train, X_test, y_test = data

    X_test_w, y_test_w, _ = window_examples(X_test, y_test, window_size, crossover)

    os.makedirs(model_path, exist_ok=True)

    print('Dataset:', X_train.shape, y_train.shape)
    model.compile(optimizer, loss=loss, metrics=['accuracy'])

    if preprocess is None:
        X_test_w_prep = X_test_w
    else:
        print('Preprocessing...')
        f, shape = preprocess
        n_samples_test = X_test_w.shape[0]
        X_test_w_prep = np.zeros((n_samples_test, *shape))

        for i in range(n_samples_test):
            X_test_w_prep[i] = f(X_test_w[i])

    generator = DataSequence(X_train, y_train, False, False, batch_size, window_size, crossover, preprocess)

    print('Training...')
    history = model.fit_generator(generator, validation_data=(X_test_w_prep, y_test_w), epochs=num_epochs,
                                  steps_per_epoch=len(generator), callbacks=callbacks, shuffle=True,
                                  use_multiprocessing=True)
    return model, optimizer, history


def train_augmented(model, optimizer, loss, name: str, num_epochs: int, batch_size: int, window_size: int,
                    crossover: float, filters: bool, salamon: bool,
                    data: Tuple[Iterable[np.ndarray], np.ndarray, Iterable[np.ndarray], np.ndarray],
                    callbacks: list = None, model_path: str = None,
                    preprocess: Optional[Tuple[Callable, Tuple]] = None):
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
    X_test_win, y_test_win, _ = window_examples(X_test, y_test, window_size, crossover)

    os.makedirs(model_path, exist_ok=True)

    print('Dataset:', X_train.shape, y_train.shape)
    model.compile(optimizer, loss=loss, metrics=['accuracy'])

    n_samples_test = X_test_win.shape[0]
    if preprocess is not None:
        f, shape = preprocess
        X_test_win_prep = np.zeros((n_samples_test, *shape))

        for i in range(n_samples_test):
            X_test_win_prep[i] = f(X_test_win[i])
    else:
        X_test_win_prep = X_test_win

    generator = DataSequence(X_train, y_train, filters, salamon, batch_size, window_size, crossover, preprocess)

    print('Training...')
    history = model.fit_generator(generator, validation_data=(X_test_win_prep, y_test_win), epochs=num_epochs,
                                  steps_per_epoch=len(generator), callbacks=callbacks, shuffle=True,
                                  use_multiprocessing=True)
    return model, optimizer, history


class DataSequence(Sequence):
    def __init__(self, X, y, filters: bool, salamon: bool, batch_size: int, window_size: int, crossover: float,
                 preprocess: Optional[Tuple[Callable, Tuple]] = None):
        self.X = X
        self.y = y
        self.dataset_size = y.shape[0]
        self.index = np.random.permutation(self.dataset_size)
        self.batch_size = batch_size
        self.window_size = window_size
        self.crossover = crossover
        self.augment_functions = []
        self.preprocess = preprocess

        if salamon:
            self.pitch_shift_1 = np.array([-2., -1., 1., 2.])
            self.pitch_shift_2 = np.array([-3.5, -2.5, 2.5, 3.5])
            self.time_stretch = np.array([0.81, 0.93, 1.07, 1.23])
            self.salamon_init()
        if filters:
            self.filter_df = None
            self.filter_init()

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
        filters = self.make_filters(n_filters, X.shape[0])
        convolved = fftconvolve(X, filters, axes=1)
        return convolved

    def salamon_init(self):
        pass

    def salamon(self, X):
        n_samples = X.shape[0]
        pitch_shift_amount = np.random.choice(self.pitch_shift_1, size=n_samples) + np.random.choice(
            self.pitch_shift_2, size=n_samples)
        stretch_amount = np.random.choice(self.time_stretch, size=n_samples)

        new_X = np.zeros_like(X)

        for i in range(n_samples):
            stretched = effects.time_stretch(X[i], stretch_amount[i])
            if stretched.size > X.shape[1]:
                stretched = stretched[:X.shape[1]]
            else:
                stretched = np.concatenate((stretched, np.zeros(X.shape[1] - stretched.size)))

            new_X[i] = effects.pitch_shift(stretched, sr=16000,
                                           n_steps=pitch_shift_amount[i])

        return new_X

    def reset(self):
        self.index = np.random.permutation(self.y.shape[0])

    def __len__(self):
        return int(np.ceil(self.dataset_size / float(self.batch_size)))

    def on_epoch_end(self):
        self.reset()

    def __getitem__(self, idx):
        try:
            index_batch = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
        except IndexError:
            index_batch = self.index[idx * self.batch_size:]
        X_batch = pad(self.X[index_batch])
        y_batch = self.y[index_batch]

        for f in self.augment_functions:
            X_batch = f(X_batch)

        # window examples
        X_win, y_win, _ = window_examples(X_batch, y_batch, self.window_size, self.crossover)

        n_samples = X_win.shape[0]
        if self.preprocess is not None:
            f, shape = self.preprocess
            X_win_prep = np.zeros((n_samples, *shape))

            for i in range(n_samples):
                X_win_prep[i] = f(X_win[i])
        else:
            X_win_prep = X_win

        # zero = np.abs(X_win_prep).sum(axis=1) < 0.01
        zero = (X_win_prep == 0).sum(axis=1) / X_win.shape[0] > 0.5
        X_win_prep = X_win_prep[~zero]
        y_win = y_win[~zero]

        return X_win_prep, y_win
