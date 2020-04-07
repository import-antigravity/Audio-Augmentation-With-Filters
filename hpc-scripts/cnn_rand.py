import sys

sys.path.append('..')
from sys import argv

# Naming convention "cnn_rand_{augm|base}_{fold}"
from tensorflow import keras

from audioaugmentation import models
from audioaugmentation.data import window_examples, dms_to_numpy
from audioaugmentation.train import train_baseline

fold = int(argv[1])
augmented = bool(int(argv[2]))

name = f'cnn_rand_{"augm" if augmented else "base"}_{fold}'

print(f'Training model cnn_rand, fold {fold}, {"not " if not augmented else ""}augmented')

X_train, y_train, X_test, y_test = dms_to_numpy(fold)

window_size = 32000
crossover = 0.5

X_train_w, y_train_w, _ = window_examples(X_train, y_train, window_size, crossover)
X_test_w, y_test_w, _ = window_examples(X_test, y_test, window_size, crossover)

kwargs = {
    'model': models.cnn_rand(),
    'optimizer': keras.optimizers.Adadelta(),
    'loss': keras.losses.mean_squared_logarithmic_error,
    'name': name,
    'num_epochs': 1000,
    'batch_size': 100,
    'data': (X_train_w, y_train_w, X_test_w, y_test_w)
}

if augmented:
    pass
else:
    train_baseline(**kwargs)
