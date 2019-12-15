import tensorflow as tf

from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

model = cnn_rand()

train(model, tf.keras.optimizers.Adam(1e-3), 1000, 10)
