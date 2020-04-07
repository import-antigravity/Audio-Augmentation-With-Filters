import tensorflow as tf

from audioaugmentation import models
from audioaugmentation.data import dms_to_numpy, conform_examples

fold = 5

model = models.cnn_rand()
model.load_weights(tf.train.latest_checkpoint(f'../models/cnn_rand_base_{fold}'))

X_train, y_train, X_test, y_test = dms_to_numpy(fold)

window_size = 32000
crossover = 0.5

X, y = conform_examples(X_test, y_test, window_size, crossover)
y_hat = model.predict(X)

accuracy = (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()

print(f"Accuracy for fold {fold}: {accuracy}")
