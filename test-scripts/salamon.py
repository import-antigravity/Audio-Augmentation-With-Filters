import sys

import librosa

sys.path.append('..')

import numpy as np
import tensorflow as tf

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy, window_examples

fold = 2
augmented = 0

model = models.salamon()
latest = tf.train.latest_checkpoint(f'../models/salamon_{augmented if augmented else "base"}_{fold}')
model.load_weights(latest)

print(model.summary())

X_train, y_train, X_test, y_test = dms_to_numpy(fold)
y = np.argmax(y_test, axis=1)

window_size = 48000
crossover = 0.5

X_win, y_win, index = window_examples(X_test, y_test, window_size, crossover)

X_win_prep = np.zeros((X_win.shape[0], 128, 128))
for i in range(X_win.shape[0]):
    win_hop = 375
    S = librosa.feature.melspectrogram(X_win[i], sr=16000, n_mels=128, n_fft=win_hop, hop_length=win_hop)
    X_win_prep[i] = S

y_hat_win = model.predict(X_win_prep)

msle = np.power(np.log(y_win + 1) - np.log(y_hat_win + 1), 2).mean()
print(f"MSLE: {msle}")

y_hat = np.zeros(X_test.shape[0])

window = 0
# Perform majority voting
for i in range(y_hat.size):
    votes = np.zeros(10)
    try:
        while index[window] == i:
            votes += y_hat_win[window]
            window += 1
        y_hat[i] = np.argmax(votes)
    except IndexError:
        pass

accuracy = (y_hat == y).mean()

print(f"Accuracy for fold {fold}: {accuracy}")
