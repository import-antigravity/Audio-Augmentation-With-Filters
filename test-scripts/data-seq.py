import sounddevice as sd
from matplotlib import pyplot as plt

from audioaugmentation.data import dms_to_numpy, classes
from audioaugmentation.train import DataSequence

X_train, y_train, X_test, y_test = dms_to_numpy(1)

print('Dataset:', X_train.shape, y_train.shape)

generator = DataSequence(X_train, y_train, False, False, 20, 16000, 0.5, None)

X, y = generator[0]

print(X.shape)
print(y.shape)

for i in range(X.shape[0]):
    plt.plot(X[i])
    plt.show()
    print(y[i])
    print(classes[y[i].argmax()])
    sd.play(X[i], samplerate=16000, blocking=True)
