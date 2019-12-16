
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import soundfile as sf

fulldatasetpath = 'UrbanSound8K/audio/'
metadata = pd.read_csv('/Users/jiax/PycharmProjects/AUDIO/UrbanSound8K/metadata/UrbanSound8K.csv')
features = []
org = []

for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))
    class_label = row["class"]
    fold  = row["fold"]
    ### Librosa
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast',sr=16000)
    features.append([audio, sample_rate, class_label,fold])

    ### Original

    audio2, sample_rate2 = sf.read(file_name)
    org.append([audio2, sample_rate2])


# Convert into a Panda dataframe
#featuresdf = pd.DataFrame(features, columns=['audio','sample_rate', 'class_label','fold'])
orgdf = pd.DataFrame(org, columns=['audio','sample_rate'])





folderpath = 'figures/'
arr = os.listdir('figures/')


audios = []
for index in arr:
    file_name = os.path.join(os.path.abspath('figures/'), index)
    print(index)
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast',sr=16000)
    audios.append(audio)
    #ipd.Audio(filename)

names = ['drilling','gun_shot','dog_bark', 'engine_idling', 'siren','children_playing',
         'street_music','car_horn','jackhammer', 'air_conditioner']

plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13

def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(10,15),dpi = 100)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,2,i)
        librosa.display.waveplot(np.array(f),sr=16000)
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.95)
    plt.tight_layout()
    plt.show()


plot_waves(names, audios)
plt.savefig('figure1.png')


#filename = 'figures/Childrenplay.wav'
#data,sample_rate = librosa.load(filename)
#_ = librosa.display.waveplot(data,sr=sample_rate)