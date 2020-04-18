import sys

import librosa

sys.path.append('..')
from sys import argv

# Naming convention "cnn_rand_{augm|base}_{fold}"
import numpy as np
from tensorflow import keras

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy
from audioaugmentation.train import train_baseline, train_augmented

fold = int(argv[1])
augmented = int(argv[2])

name = f'salamon_{augmented if augmented else "base"}_{fold}'

print(f'Training model salamon, fold {fold}, {"not " if not augmented else ""}augmented')

X_train, y_train, X_test, y_test = dms_to_numpy(fold)


def to_spectrogram(X):
    win_hop = 375
    S = librosa.feature.melspectrogram(X, sr=16000, n_mels=128, n_fft=win_hop, hop_length=win_hop)
    S = librosa.power_to_db(S, ref=np.max)  # log-scaled
    return S


kwargs = {
    'model': models.salamon(),
    'optimizer': keras.optimizers.SGD(lr=0.01),
    'loss': keras.losses.categorical_crossentropy,
    'name': name,
    'num_epochs': 200,
    'batch_size': 100,
    'window_size': 48000,
    'crossover': 0.5,
    'data': (X_train, y_train, X_test, y_test),
    'preprocess': (to_spectrogram, (128, 128))
}

if augmented:
    kwargs['batch_size'] = 20

if augmented == 1:
    # Augment with filters
    train_augmented(**kwargs, filters=True, salamon=False)
elif augmented == 2:
    # Augment from Salamon (2016)
    train_augmented(**kwargs, filters=False, salamon=True)
elif augmented == 3:
    # Use both
    train_augmented(**kwargs, filters=True, salamon=True)
else:
    train_baseline(**kwargs)
