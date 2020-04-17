import pickle
from typing import Tuple, List, Iterable

import numpy as np
import pandas as pd

classes = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]


def dms_to_numpy_old(fold: int, path: str = '../data/UrbanSound_sr16000.dms') \
        -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    with open(path, 'rb') as fp:
        itemlist = pickle.load(fp)

    itemlist.class_label.replace(
        {
            'air_conditioner': 0,
            'car_horn': 1,
            'children_playing': 2,
            'dog_bark': 3,
            'drilling': 4,
            'engine_idling': 5,
            'gun_shot': 6,
            'jackhammer': 7,
            'siren': 8,
            'street_music': 9
        },
        inplace=True
    )

    train = itemlist[itemlist.fold != fold]
    X_train = train.audio.values
    y_train = np.eye(10)[train.class_label.values]

    test = itemlist[itemlist.fold == fold]
    X_test = test.audio.values
    y_test = np.eye(10)[test.class_label.values]

    return X_train, y_train, X_test, y_test


def dms_to_numpy(fold: int, path: str = '../data/UrbanSound8K16') \
        -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    itemlist = pd.read_pickle(path)

    itemlist.class_label.replace(
        {
            'air_conditioner': 0,
            'car_horn': 1,
            'children_playing': 2,
            'dog_bark': 3,
            'drilling': 4,
            'engine_idling': 5,
            'gun_shot': 6,
            'jackhammer': 7,
            'siren': 8,
            'street_music': 9
        },
        inplace=True
    )

    train = itemlist[itemlist.fold != fold]
    X_train = train.audio.values.tolist()
    y_train = np.eye(10)[train.class_label.values]

    test = itemlist[itemlist.fold == fold]
    X_test = test.audio.values.tolist()
    y_test = np.eye(10)[test.class_label.values]

    return X_train, y_train, X_test, y_test


def pad(X_list: Iterable[np.ndarray]):
    if len(X_list) > 0:
        lengths = [a.size for a in X_list]
        X = np.zeros((len(X_list), max(lengths)))
        for i in range(len(X_list)):
            X[i, :lengths[i]] = X_list[i]
        return X
    else:
        return np.array([])


def window_examples(X_list: List[np.ndarray], y_original: np.ndarray, window_size: int, crossover: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # INPUT MUST BE NUMPY ARRAYS, THERE IS NO POINT IN CONVERTING TO TENSORS BEFORE CALLING THIS METHOD
    assert len(X_list) == y_original.shape[0]

    X = []
    y = []
    index = []

    for i in range(len(X_list)):
        sample = X_list[i].astype('float32')
        if sample.shape[0] <= window_size:
            zeros = np.zeros(window_size - sample.shape[0])
            padded = np.concatenate((sample, zeros))
            X.append(padded)
            y.append(y_original[i])
            index.append(i)
        else:
            current_start = 0
            while current_start < sample.shape[0]:
                zeros = np.zeros(0)
                if current_start + window_size > sample.shape[0]:
                    zeros = np.zeros(current_start + window_size - sample.shape[0])
                padded = np.concatenate((sample[current_start:current_start + window_size], zeros))
                X.append(padded)
                y.append(y_original[i])
                index.append(i)
                current_start += int((1. - crossover) * window_size)

    return np.vstack(X), np.vstack(y), np.array(index)
