import numpy as np
import sounddevice as sd

from audioaugmentation.data import dms_to_numpy
from audioaugmentation.train import DataSequence

dg = DataSequence(np.zeros(1), np.zeros(1), True, True, 10, 32000, 0.5)

X = np.vstack((dms_to_numpy(1)[0][0].reshape(1, -1),) * 10)
signals = dg.salamon(X)

for i in range(X.shape[0]):
    sd.play(signals[i], blocking=True, samplerate=16000)
