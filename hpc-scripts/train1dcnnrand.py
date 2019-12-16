import sys
sys.path.append('..')

import tensorflow as tf

from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

model = cnn_rand()

print(model.summary())

classifier, optimizer, history = train(model,
                                       optimizer=tf.keras.optimizers.Adam(1e-3),
                                       epochs=1000,
                                       batch_size=10,
                                       path='../models/1DCNNRand32k/',
                                       feature_size=32000,
                                       num_gpus=2)
