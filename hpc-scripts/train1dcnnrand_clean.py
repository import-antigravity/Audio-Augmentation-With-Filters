import sys
sys.path.append('..')

import tensorflow as tf

from audioaugmentation.data import import_clean_data
from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

model = cnn_rand()

print(model.summary())

data = import_clean_data('../data/', 32000)

classifier, optimizer, history = train(data, model,
                                       optimizer=tf.keras.optimizers.Adam(1e-3),
                                       epochs=1000,
                                       batch_size=10,
                                       path=str(sys.argv[2]),
                                       num_gpus=int(sys.argv[1]))
