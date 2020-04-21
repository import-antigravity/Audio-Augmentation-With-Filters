import sys

import librosa

sys.path.append('..')

import numpy as np
import pandas as pd
import tensorflow as tf
from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy, window_examples

data = {
    'model': [],
    'accuracy': [],
    'fold': [],
    'augmented': [],
    'label': []
}

crossover = 0.5

for m, model in [('cnn_rand16k', models.cnn_rand16k), ('salamon', models.salamon)]:
    model = model()
    print(model.summary())

    if m == 'cnn_rand16k':
        window_size = 16000


        def preprocess(X):
            return X
    else:
        window_size = 48000


        def preprocess(X):
            X_new = np.zeros((X.shape[0], 128, 128))
            for i in range(X.shape[0]):
                win_hop = 375
                S = librosa.feature.melspectrogram(X[i], sr=16000, n_mels=128, n_fft=win_hop, hop_length=win_hop)
                S = librosa.power_to_db(S, ref=np.max)  # log-scaled
                X_new[i] = S
            return X_new

    for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for augmented in [0, 1, 2, 3]:
            for label in ['all', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                latest = tf.train.latest_checkpoint(f'../models/{m}_{augmented if augmented else "base"}_{fold}')
                model.load_weights(latest)

                _, _, X_test, y_test = dms_to_numpy(fold)
                y = np.argmax(y_test, axis=1)

                if label != 'all':
                    this_label = y == label
                    X_test = X_test[this_label]
                    y_test = y_test[this_label]
                    y = y[this_label]

                X_win, y_win, index = window_examples(X_test, y_test, window_size, crossover)

                X_test_p = preprocess(X_win)
                y_hat_win = model.predict(X_test_p)

                msle = np.power(np.log(y_win + 1) - np.log(y_hat_win + 1), 2).mean()
                print(f"MSLE: {msle}")

                y_hat = np.zeros(y_test.shape[0])

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

                print(f"{m} {augmented} accuracy for fold {fold}, class label {label}: {accuracy}")

                data['model'].append(m)
                data['accuracy'].append(accuracy)
                data['fold'].append(fold)
                data['augmented'].append(augmented)
                data['label'].append(label)

df = pd.DataFrame(data)
df.to_csv('results.csv')
