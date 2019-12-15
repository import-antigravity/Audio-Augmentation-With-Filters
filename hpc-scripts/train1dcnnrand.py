import sys
sys.path.append('..')

import tensorflow as tf
import pickle

from audioaugmentation.models import cnn_rand
from audioaugmentation.train import train

model = cnn_rand()

print(model.summary())

classifier, optimizer, history = train(model, tf.keras.optimizers.Adam(1e-3), 1000, 10)

pickle.dump(classifier, '1dcnnrand')
pickle.dump(history, 'history')
