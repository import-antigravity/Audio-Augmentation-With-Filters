import tensorflow as tf


def cnn_rand32k():
    cnn = [
        tf.keras.layers.InputLayer(input_shape=(32000,)),
        tf.keras.layers.Reshape((32000, 1)),
        tf.keras.layers.Conv1D(16, 64, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
        tf.keras.layers.Conv1D(32, 32, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
        tf.keras.layers.Conv1D(64, 16, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 8, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(256, 4, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=4, strides=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]

    return tf.keras.Sequential(cnn)


def cnn_rand16k():
    cnn = [
        tf.keras.layers.InputLayer(input_shape=(16000,)),
        tf.keras.layers.Reshape((16000, 1)),
        tf.keras.layers.Conv1D(16, 64, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
        tf.keras.layers.Conv1D(32, 32, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
        tf.keras.layers.Conv1D(64, 16, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 8, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ]

    return tf.keras.Sequential(cnn)
