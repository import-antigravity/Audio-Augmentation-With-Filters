import sounddevice as sd

from audioaugmentation.data import dms_to_numpy, classes

X_train, y_train, X_test, y_test = dms_to_numpy(1)

print(len(X_train), y_train.shape, len(X_test), y_test.shape)

for i in range(len(X_train)):
    print(classes[y_train[i].argmax()])
    sd.play(X_train[i], samplerate=16000, blocking=True)
