import librosa
import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

from audioaugmentation.data import dms_to_numpy, window_examples

X_train, y_train, X_test, y_test = dms_to_numpy(1)
X, _, _ = window_examples(X_train, y_train, 16000 * 3, 0.5)

i = 32

sample = X[i]
print(sample.shape)

win_hop = 375
S = librosa.feature.melspectrogram(sample, sr=16000, n_mels=128, n_fft=win_hop, hop_length=win_hop)
print(S.shape)

S_dB = librosa.power_to_db(S, ref=np.max)

plt.imshow(S_dB)
plt.show()

sd.play(sample, samplerate=16000, blocking=True)
