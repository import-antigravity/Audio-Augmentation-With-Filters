import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf
from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy, window_examples

fold = 2
augmented = 0

model = models.cnn_rand32k()
latest = tf.train.latest_checkpoint(f'../models/cnn_rand_{augmented if augmented else "base"}_{fold}')
model.load_weights(latest)

print(model.summary())

X_train, y_train, X_test, y_test = dms_to_numpy(fold)
y = np.argmax(y_test, axis=1)

window_size = 32000
crossover = 0.5

X_win, y_win, index = window_examples(X_test, y_test, window_size, crossover)
y_hat_win = model.predict(X_win)

msle = np.power(np.log(y_win + 1) - np.log(y_hat_win + 1), 2).mean()
print(f"MSLE: {msle}")

y_hat = np.zeros(X_test.shape[0])

window = 0
# Perform majority voting
for i in range(y_hat.size):
    votes = np.zeros(10)
    try:
        while index[window] == i:
            # Don't count windows where there isn't enough information
            percent_zero = (X_win[window] == 0).sum() / X_win[window].size
            '''
            if percent_zero > 0.5:
                print(window)
                print(classes[y_win[window].argmax()])
                plt.plot(X_win[window])
                plt.show()
                sd.play(X_win[window], blocking=True, samplerate=16000)
            '''
            if percent_zero <= 0.5:
                votes += y_hat_win[window]
            window += 1
        y_hat[i] = np.argmax(votes)
    except IndexError:
        pass

accuracy = (y_hat == y).mean()

print(f"Accuracy for fold {fold}: {accuracy}")
