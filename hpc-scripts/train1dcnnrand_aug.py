import sys

sys.path.append('..')

import tensorflow as tf

from audioaugmentation.data import import_augmented_data
from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

tf.logging.set_verbosity(tf.logging.ERROR)

model = cnn_rand()

print(model.summary())

data = import_augmented_data('../data', 32000, int(sys.argv[3]), 0., 0., 100)

classifier, optimizer, history = train(data, model,
                                       optimizer=tf.keras.optimizers.Adam(1e-3),
                                       epochs=500,
                                       batch_size=10,
                                       model_path=str(sys.argv[2]),
                                       num_gpus=int(sys.argv[1]))
