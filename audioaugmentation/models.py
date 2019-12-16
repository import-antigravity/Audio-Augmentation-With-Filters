import tensorflow as tf


def cnn_rand():
    cnn = [
        tf.keras.layers.InputLayer(input_shape=(32000,)),
        tf.keras.layers.Reshape((32000, 1)),
        tf.keras.layers.Conv1D(16, 64, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
        tf.keras.layers.Conv1D(32, 32, strides=2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=8),
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
