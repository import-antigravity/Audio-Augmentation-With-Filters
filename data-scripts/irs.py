import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

files = os.listdir('../data/IRs/outdoor')

data = []

irs = 0

for file in files:
    if '.wav' in file:
        original, sr = sf.read(os.path.join('../data/IRs/outdoor', file))

        for channel in range(original.shape[1]):
            mono = original[:, channel]
            resampled = librosa.resample(np.asfortranarray(mono), sr, 16000)
            resampled = resampled[np.abs(resampled).argmax() - 10:]
            data.append(resampled)

df = pd.DataFrame(data)
pd.to_pickle(df, '../data/IR16outdoor')
