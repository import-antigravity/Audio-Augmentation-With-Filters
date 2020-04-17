import librosa
import numpy as np
import pandas as pd
import soundfile as sf

meta = pd.read_csv('../data/UrbanSound8K/metadata/UrbanSound8K.csv')

d = {
    'audio': [],
    'fold': [],
    'class_label': []
}

for i, m in meta.iterrows():
    path = f"../data/UrbanSound8K/audio/fold{m['fold']}/{m['slice_file_name']}"
    class_label = m['class']
    y, sr = sf.read(path)
    if len(y.shape) > 1:
        y = y.sum(axis=1)
    resampled = librosa.core.resample(np.asfortranarray(y), sr, 16000)
    d['audio'].append(resampled)
    d['fold'].append(m['fold'])
    d['class_label'].append(class_label)

new_df = pd.DataFrame(d)
new_df.to_pickle('../data/UrbanSound8K16')
