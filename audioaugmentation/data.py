import numpy as np
import tensorflow as tf
import pickle
import pandas
from sklearn.preprocessing import LabelEncoder
import pyroomacoustics as pra


def import_dataset():
    with open('..\\data\\UrbanSound_sr16000', 'rb') as fp:
        itemlist = pickle.load(fp)
    print(itemlist.columns)
    labels = itemlist.pop('class_label')
    print(itemlist.columns)
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

def import_augmented_data(augmentation_precent: float, noise_mean: float, noise_stddev: float, num_rooms: int):
    with open('..\\data\\UrbanSound_sr16000', 'rb') as fp:
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
        test_features.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > (1-augmentation_precent), lambda: f(x), lambda: x))
        train_features.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > (1-augmentation_precent), lambda: f(x), lambda: x))
    test_features, test_labels = conform_examples(list(test_features), test_labels, 50999, 0.5)
    train_features, train_labels = conform_examples(list(train_features), train_labels, 50999, 0.5)
    Testing_Dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    Training_Dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    return Training_Dataset, Testing_Dataset


def conform_examples(X_list: [np.ndarray], y_original: np.ndarray, window_size: int, crossover: float):
    assert len(X_list) == len(y_original)

    X = []
    y = []

    for i in range(len(X_list)):
        sample = X_list[i]
        if sample.size <= window_size:
            zeros = np.zeros(window_size - sample.size)
            padded = np.concatenate((sample, zeros))
            X.append(padded)
            y.append(y_original[i])
        else:
            current_start = 0
            while current_start < sample.size:
                zeros = np.array([])
                if current_start + window_size > sample.size:
                    zeros = np.zeros(current_start + window_size - sample.size)
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
        noise = np.random.normal(shape=x.shape, mean=self.mean, stddev=self.stddev)
        return x + noise


class room_distribution(object):

    def __init__(self, num_rooms):
        self.rooms = []
        # TODO: create a distribution of PRA rooms
        for i in range(num_rooms):
            dims = np.random.randint(low=1, high=10, size=2)
            room = pra.ShoeBox(dims, fs=16000)
            self.rooms.append(room)

    def sample_source(self, room: pra.Room):
        walls = room.walls
        for wall in walls:
            if wall.corners[0][0] == wall.corners[1][0]:
                y_len = np.abs(wall.corners[0][1] - wall.corners[1][1])
            else:
                x_len = np.abs(wall.corners[0][0] - wall.corners[1][0])
        y_pos = np.random.uniform(low=0, high=y_len)
        x_pos = np.random.uniform(low=0, high=x_len)
        d = []
        d.append(x_pos)
        d.append(y_pos)
        d = np.asarray(d)
        return pra.SoundSource(position=d)

    def sample_mic(self, room: pra.Room):

        pass

    # returns a room populated with a source and microphone array, drawn from the random distributions
    def sample(self):
        # TODO: add a random source and microphone to a random room, then return
        room = np.random.choice(self.rooms)
        source = self.sample_source(room)
        print(source.dim)
        print(np.asarray(source.position).shape[1])
        mic = self.sample_mic(room)
        room.add_source(source)
        room.add_microphone_array(mic)
        return room

    def augment(self, x: tf.Tensor) -> tf.Tensor:
        room = self.sample()
        room.sources[0].add_signal(x)
        room.simulate()
        return room.mic_array.signals[1, :]


A, b = import_dataset()