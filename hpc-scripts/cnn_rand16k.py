import sys

sys.path.append('..')
from sys import argv

# Naming convention "cnn_rand_{augm|base}_{fold}"
from tensorflow import keras
import numpy as np
import pandas as pd

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy
from audioaugmentation.train import train_baseline, train_augmented

fold = int(argv[1])
augmented = int(argv[2])

name = f'cnn_rand16k_{augmented if augmented else "base"}_{fold}'

print(f'Training model cnn_rand, fold {fold}, {"not " if not augmented else ""}augmented')

X_train, y_train, X_test, y_test = dms_to_numpy(fold)

print(X_train.dtype)

# Limit model
'''
idx = np.random.permutation(X_train.shape[0])
X_train = np.concatenate((X_train, X_test), axis=0) 
y_train = np.concatenate((y_train, y_test), axis=0)
X_test = X_test
y_test = y_test
'''

kwargs = {
    'model': models.cnn_rand16k(),
    'optimizer': keras.optimizers.Adadelta(),
    'loss': keras.losses.mean_squared_logarithmic_error,
    'name': name,
    'num_epochs': 200,
    'batch_size': 10,
    'window_size': 16000,
    'crossover': 0.5,
    'data': (X_train, y_train, X_test, y_test)
}

if augmented:
    kwargs['batch_size'] = 10

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
