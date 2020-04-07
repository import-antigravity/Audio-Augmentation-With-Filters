import numpy as np
import sounddevice as sd

from audioaugmentation.data import dms_to_numpy, window_examples

X_train, y_train, X_test, y_test = dms_to_numpy(1)

window_size = 32000
crossover = 0.5

X_conform, y_conform, index = window_examples(X_train, y_train, window_size, crossover)

print(X_train[7].shape)
# sd.play(X_train[7] / np.abs(X_train[7]).max(), blocking=True, samplerate=16000)


for i in range(200, 300):
    print([
              'air_conditioner',
              'car_horn',
              'children_playing',
              'dog_bark',
              'drilling',
              'engine_idling',
              'gun_shot',
              'jackhammer',
              'siren',
              'street_music'
          ][y_conform.argmax(axis=1)[i]])
    sd.play(X_conform[i] / np.abs(X_conform[i]).max(), blocking=True, samplerate=16000)
