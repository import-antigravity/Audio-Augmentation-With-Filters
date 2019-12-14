import numpy as np
import tensorflow as tf
import pickle
import pandas
from sklearn.preprocessing import LabelEncoder
import pyroomacoustics as pra


def import_dataset():
    with open('data\UrbanSound8KDataFrame', 'rb') as fp:
        itemlist = pickle.load(fp)
    print(itemlist.columns)
    labels = itemlist.pop('class_label')
    print(itemlist.columns)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = tf.one_hot(labels, 10)
    features = itemlist.pop('audio')
    print(len(labels))
    print(len(features))
    print(type(labels))
    features, labels = conform_examples(features, labels, 1000, 0.5)
    features = tf.convert_to_tensor(features, np.float32)
    UrbanSound = tf.data.Dataset.from_tensor_slices((features, labels))
    test_set_size = int(0.2 * features.shape[0])
    Testing_Dataset = UrbanSound.take(test_set_size)
    Training_Dataset = UrbanSound.skip(test_set_size)
    return Training_Dataset, Testing_Dataset


def conform_examples(X_list: [np.ndarray], y_original: np.ndarray, window_size: int, crossover: float):
    assert len(X_list) == y_original.size

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
