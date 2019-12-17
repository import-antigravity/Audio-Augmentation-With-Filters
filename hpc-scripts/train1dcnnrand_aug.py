import sys

import numpy as np

sys.path.append('..')

import tensorflow as tf

from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

tf.logging.set_verbosity(tf.logging.ERROR)

model = cnn_rand()

print(model.summary())

npz = np.load('../data/augmented.npz')

test_features_name, test_labels_name, train_features_name, train_labels_name = npz.files

data = npz[test_features_name], npz[test_labels_name], npz[train_features_name], npz[train_labels_name]

classifier, optimizer, history = train(data, model,
                                       optimizer=tf.keras.optimizers.Adam(1e-3),
                                       epochs=500,
                                       batch_size=10,
                                       model_path=str(sys.argv[2]),
                                       num_gpus=int(sys.argv[1]))
