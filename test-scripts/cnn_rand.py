import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy, window_examples

fold = 1

model = models.cnn_rand32k()
model.load_weights(tf.train.latest_checkpoint(f'../models/cnn_rand_base_{fold}'))

print(model.summary())

X_train, y_train, X_test, y_test = dms_to_numpy(fold)
y = y_test.argmax(axis=1)

window_size = 32000
crossover = 0.5

X_win, y_win, index = window_examples(X_test, y_test, window_size, crossover)
y_hat_win = model.predict(X_win)

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
