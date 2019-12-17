import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from sklearn.preprocessing import LabelEncoder


def data_frame_to_folds(data_path: str):
    with open(os.path.join(data_path, 'UrbanSound_sr16000'), 'rb') as fp:
        itemlist = pickle.load(fp)
    labels = itemlist.pop('class_label')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np.eye(10)[labels]
    features = itemlist.pop('audio')
    folds = itemlist.pop('fold')
    data_folds = []
    label_folds = []
    for i in range(10):
        fold_features = features[np.where(folds == i + 1)[0]].tolist()
        fold_labels = labels[np.where(folds == i + 1)[0]]
        data_folds.append(fold_features)
        label_folds.append(fold_labels)
    # data_folds: list of lists, label_folds: list of arrays
    return data_folds, label_folds


def combine_folds(data_folds, label_folds):
    test_features = data_folds[-1]
    train_features = [example for fold in data_folds[:-1] for example in fold]
    test_labels = label_folds[-1]
    train_labels = np.concatenate(label_folds[:-1])

    # features are lists, labels are arrays
    return test_features, test_labels, train_features, train_labels


def import_clean_data(data_path: str, feature_size: int, crossover: float = 0.5):
    data_folds, label_folds = data_frame_to_folds(data_path)
    test_features, test_labels, train_features, train_labels = combine_folds(data_folds, label_folds)

    # Conform
    train_features, train_labels = conform_examples(train_features, train_labels, feature_size, crossover)
    test_features, test_labels = conform_examples(test_features, test_labels, feature_size, crossover)

    return test_features, test_labels, train_features, train_labels


def import_augmented_data(data_path: str, feature_size: int, augmentation_factor: int, noise_mean: float,
                          noise_stddev: float, num_rooms: int, crossover: float = 0.5):
    data_folds, label_folds = data_frame_to_folds(data_path)
    test_features, test_labels, train_features, train_labels = combine_folds(data_folds, label_folds)

    # Augment
    nd = noise_distribution(noise_mean, noise_stddev)
    rd = room_distribution(num_rooms)

    new_features = []
    new_labels = []

    print("Applying augmentation:")
    for i in range(augmentation_factor - 1):
        for j, clip in enumerate(train_features):
            transformed = nd.augment(rd.augment(clip))
            new_features.append(transformed)
            new_labels.append(train_labels[j])

    train_features += new_features
    train_labels = np.concatenate((train_labels, np.array(new_labels)))

    # Conform
    train_features, train_labels = conform_examples(train_features, train_labels, feature_size, crossover)
    test_features, test_labels = conform_examples(test_features, test_labels, feature_size, crossover)

    return test_features, test_labels, train_features, train_labels


def conform_examples(X_list: [np.ndarray], y_original: np.ndarray, window_size: int, crossover: float):
    # INPUT MUST BE NUMPY ARRAYS, THERE IS NO POINT IN CONVERTING TO TENSORS BEFORE CALLING THIS METHOD
    assert len(X_list) == y_original.shape[0]

    X = []
    y = []

    for i in range(len(X_list)):
        sample = X_list[i].astype('float32')
        print(np.abs(sample).max())
        sample = sample / np.abs(sample).max()  # Normalize samples
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


class noise_distribution(object):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def augment(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=self.mean, scale=self.stddev, size=x.shape)
        return x + noise


class room_distribution(object):

    def __init__(self, num_rooms):
        self.rooms = []
        # TODO: create a distribution of PRA rooms
        for i in range(num_rooms):
            dims = np.random.randint(low=1, high=10, size=2)
            self.rooms.append(dims)

    @staticmethod
    def sample_loc(walls):
        for wall in walls:
            if wall.corners[0][0] == wall.corners[1][0]:
                x_len = np.abs(wall.corners[0][1] - wall.corners[1][1])
            else:
                y_len = np.abs(wall.corners[0][0] - wall.corners[1][0])
        y_pos = np.random.uniform(low=0, high=y_len)
        x_pos = np.random.uniform(low=0, high=x_len)
        d = []
        d.append(x_pos)
        d.append(y_pos)
        # d = np.asarray(d)
        return d

    @staticmethod
    def sample_source(room: pra.Room):
        d = room_distribution.sample_loc(room.walls)
        return d

    @staticmethod
    def sample_mic(room: pra.Room):
        d = room_distribution.sample_loc(room.walls)
        R = pra.linear_2D_array(d, 1, 0, 0.04)
        R = pra.MicrophoneArray(R, fs=16000)
        return R

    # returns a room populated with a source and microphone array, drawn from the random distributions
    def sample(self):
        # TODO: add a random source and microphone to a random room, then return
        dims = random.choice(self.rooms)
        room = pra.ShoeBox(dims, fs=16000)
        source = room_distribution.sample_source(room)
        mic = room_distribution.sample_mic(room)
        room.add_source(source)
        room.add_microphone_array(mic)
        return room

    def augment(self, x: np.ndarray) -> np.ndarray:
        room = self.sample()
        room.sources[0].add_signal(x)
        room.compute_rir()
        ir = room.rir[0][0]
        signal = np.convolve(x, ir)
        return signal


if __name__ == "__main__":
    a, b, c, d = import_augmented_data("..\\data", 50999, 3, 0, 1, 100)