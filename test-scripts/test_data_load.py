from audioaugmentation.data import dms_to_numpy, window_examples

X_train, y_train, X_test, y_test = dms_to_numpy(1)

window_size = 32000
crossover = 0.5

X_conform, y_conform, index = window_examples(X_train, y_train, window_size, crossover)

print(X_train[7].shape)
# sd.play(X_train[7] / np.abs(X_train[7]).max(), blocking=True, samplerate=16000)

