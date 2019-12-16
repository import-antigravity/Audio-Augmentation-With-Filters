import os
import numpy as np
import tensorflow as tf
import pickle
import pandas
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def import_dataset(data_path: str):
    with open(data_path + os.sep + 'UrbanSound_sr16000.dms', 'rb') as fp:
        itemlist = pickle.load(fp)
    labels = itemlist.pop('class_label')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = tf.one_hot(labels, 10)
    features = itemlist.pop('audio')
    test_set_size = int(0.2 * features.shape[0])
    test_features = features[:test_set_size]
    train_features = features[test_set_size:]
    test_labels = labels[:test_set_size]
    train_labels = labels[test_set_size:]
    test_features, test_labels = conform_examples(list(test_features), test_labels, 50999, 0.5)
    train_features, train_labels = conform_examples(list(train_features), train_labels, 50999, 0.5)
    Testing_Dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    Training_Dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    return Training_Dataset, Testing_Dataset


def import_numpy(data_path: str, feature_size: int):
    with open(data_path + os.sep + 'UrbanSound_sr16000.dms', 'rb') as fp:
        itemlist = pickle.load(fp)
    labels = itemlist.pop('class_label')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np.eye(10)[labels]
    features = itemlist.pop('audio')
    test_set_size = int(0.2 * features.shape[0])
    test_features = features[:test_set_size]
    train_features = features[test_set_size:]
    test_labels = labels[:test_set_size]
    train_labels = labels[test_set_size:]
    test_features, test_labels = conform_examples(list(test_features), test_labels, feature_size, 0.5)
    train_features, train_labels = conform_examples(list(train_features), train_labels, feature_size, 0.5)
    return test_features, test_labels, train_features, train_labels


def import_augmented_data(augmentation_percent: float, noise_mean: float, noise_stddev: float, num_rooms: int):
    with open(f'..{os.sep}data{os.sep}UrbanSound_sr16000.dms', 'rb') as fp:
        itemlist = pickle.load(fp)
    labels = itemlist.pop('class_label')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = tf.one_hot(labels, 10)
    features = itemlist.pop('audio')
    nd = noise_distribution(noise_mean, noise_stddev)
    rd = room_distribution(num_rooms)
    test_set_size = int(0.2 * features.shape[0])
    test_features = features[:test_set_size]
    train_features = features[test_set_size:]
    test_labels = labels[:test_set_size]
    train_labels = labels[test_set_size:]
    for f in [rd.augment, nd.augment]:
        test_features.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > (1 - augmentation_percent), lambda: f(x), lambda: x))
        train_features.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > (1 - augmentation_percent), lambda: f(x), lambda: x))
    test_features, test_labels = conform_examples(list(test_features), test_labels, 50999, 0.5)
    train_features, train_labels = conform_examples(list(train_features), train_labels, 50999, 0.5)
    Testing_Dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    Training_Dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    return Training_Dataset, Testing_Dataset


def conform_examples(X_list: [np.ndarray], y_original: np.ndarray, window_size: int, crossover: float):
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

class noise_distribution(object):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def augment(self, x: tf.Tensor) -> tf.Tensor:
        noise = np.random.normal(loc=self.mean, scale=self.stddev, size=x.shape)
        return x + noise


class room_distribution(object):

    def __init__(self, num_rooms):
        self.rooms = []
        # TODO: create a distribution of PRA rooms
        for i in range(num_rooms):
            dims = np.random.randint(low=1, high=10, size=2)
            room = pra.ShoeBox(dims, fs=16000)
            self.rooms.append(room)

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
        #d = np.asarray(d)
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
        room = np.random.choice(self.rooms)
        source = room_distribution.sample_source(room)
        mic = room_distribution.sample_mic(room)
        room.add_source(source)
        room.add_microphone_array(mic)
        return room

    def augment(self, x: tf.Tensor) -> tf.Tensor:
        room = self.sample()
        room.sources[0].add_signal(x)
        room.simulate()
        return room.mic_array.signals[1, :]


if __name__ == "__main__":
    print(np.array([2.4238, 4.43235]))
    rd = room_distribution(100)
    s = rd.sample()
    s.plot()
    plt.show()
