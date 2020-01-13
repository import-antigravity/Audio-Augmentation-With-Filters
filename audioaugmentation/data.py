import pickle
from typing import Tuple, List

import numpy as np


def dms_to_numpy(fold: int, path: str = '../data/UrbanSound_sr16000.dms') \
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


def conform_examples(X_list: List[np.ndarray], y_original: np.ndarray, window_size: int, crossover: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    # INPUT MUST BE NUMPY ARRAYS, THERE IS NO POINT IN CONVERTING TO TENSORS BEFORE CALLING THIS METHOD
    assert len(X_list) == y_original.shape[0]

    X = []
    y = []

    for i in range(len(X_list)):
        sample = X_list[i].astype('float32')
        if sample.shape[0] <= window_size:
            zeros = np.zeros(window_size - sample.shape[0])
            padded = np.concatenate((sample, zeros))
            X.append(padded)
            y.append(y_original[i])
        else:
            current_start = 0
            while current_start < sample.shape[0]:
                zeros = np.zeros(0)
                if current_start + window_size > sample.shape[0]:
                    zeros = np.zeros(current_start + window_size - sample.shape[0])
                padded = np.concatenate((sample[current_start:current_start + window_size], zeros))
                X.append(padded)
                y.append(y_original[i])
                current_start += int((1. - crossover) * window_size)

    return np.vstack(X), np.vstack(y)
