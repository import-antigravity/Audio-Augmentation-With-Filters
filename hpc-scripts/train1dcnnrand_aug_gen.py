import sys

sys.path.append('..')

import tensorflow as tf

from audioaugmentation.data import import_augmented_data_gen
from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train_generator

tf.logging.set_verbosity(tf.logging.ERROR)

model = cnn_rand()

print(model.summary())

data = import_augmented_data_gen('../data', 32000, 0., int(sys.argv[3]), 0., 300, 10)

classifier, optimizer, history = train_generator(data, model, optimizer=tf.keras.optimizers.Adam(1e-3), epochs=500,
                                                 model_path=str(sys.argv[2]), num_gpus=int(sys.argv[1]))
