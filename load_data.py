import os
import sys
import numpy as np 
import pandas as pd
import data_preprocess as dp

audio_folders  = './data'
nationalities = ['brazil', 'french', 'chinese', 'canadian']

def get_melspectrograms(nat):
    dirr = os.path.join(audio_folders, nat)
    files = os.listdir(dirr)
    spectrograms = np.asarray([dp.log_scale_melspectrogram(os.path.join(audio_folders, nat, i)) for i in files])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms

if __name__ == '__main__':
    f = os.listdir(audio_folders)
    nat_folders = [i for i in f if os.path.isdir(os.path.join(audio_folders, i))]
    spec_list = []
    for nat in nat_folders:
        spec_list.append(get_melspectrograms(nat))
    spectrograms = np.concatenate(spec_list)
    
    print(spectrograms.shape)
    print(spectrograms[0].shape)
    print(spectrograms[1].shape)