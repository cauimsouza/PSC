import os

import numpy as np

import data_preprocess as dp

dir_path = os.path.dirname(os.path.realpath(__file__))
audio_folders  = os.path.join(dir_path, 'datasets')
spec_folder = os.path.join(dir_path, 'spectrograms')

def get_melspectrograms(path, nat):
    dirr = os.path.join(path, nat)
    files = os.listdir(dirr)
    spectrograms = np.asarray([dp.log_scale_melspectrogram(os.path.join(dirr, i)) for i in files])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2])
    print(spectrograms.shape)
    return spectrograms

if __name__ == '__main__':
    phrases = os.listdir(audio_folders)
    for phrase in phrases:
        print("Folder {}".format(phrase))
        path = os.path.join(audio_folders, phrase)
        nat_folders = os.listdir(path)
        for nat in nat_folders:
            print("Constructing {}..".format(nat))
            spec_list = []
            spec_list.append(get_melspectrograms(path, nat))
            print("Saving {} spectrograms..".format(nat))
            specs = np.concatenate(spec_list)
            np.save(os.path.join(spec_folder, phrase, '{}spec.npy'.format(nat)), specs)
        print("")