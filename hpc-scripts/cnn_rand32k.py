import sys

sys.path.append('..')
from sys import argv

# Naming convention "cnn_rand_{augm|base}_{fold}"
from tensorflow import keras

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy
from audioaugmentation.train import train_baseline, train_augmented

fold = int(argv[1])
augmented = int(argv[2])

name = f'cnn_rand32k_{augmented if augmented else "base"}_{fold}'

print(f'Training model cnn_rand, fold {fold}, {"not " if not augmented else ""}augmented')

X_train, y_train, X_test, y_test = dms_to_numpy(fold)

kwargs = {
    'model': models.cnn_rand32k(),
    'optimizer': keras.optimizers.Adadelta(),
    'loss': keras.losses.mean_squared_logarithmic_error,
    'name': name,
    'num_epochs': 200,
    'batch_size': 100,
    'window_size': 32000,
    'crossover': 0.5,
    'data': (X_train, y_train, X_test, y_test)
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
