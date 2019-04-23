import srt
import shutil
import numpy as np
import datetime
import os
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from joblib import dump, load
from sklearn.preprocessing import scale

def get_speech_features(audio_file, log=False):
    frequency_sampling, audio_signal = wavfile.read(audio_file)
    # audio_signal = audio_signal[:1500000]

    features_mfcc = mfcc(signal=audio_signal, samplerate=frequency_sampling, winlen=0.025, nfft=512)

    features_mfcc = features_mfcc.T

    filterbank_features = logfbank(signal=audio_signal, samplerate=frequency_sampling, winlen=0.025, nfft=512)

    if log:
        print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
        print('Length of each feature =', features_mfcc.shape[1])
        print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
        print('Length of each feature =', filterbank_features.shape[1])

    return scale(np.array(filterbank_features), axis=0, with_mean=True, with_std=True, copy=True)

def get_subtitles_presence_array(srt_file, length, offset=0, padding=0):
    with open(srt_file) as file:
        srt_string = file.read()
        subtitles = list(srt.parse(srt_string))
        subtitles_presence_array = np.zeros(int(subtitles[-1].end.total_seconds()) + offset)

        # generate subtitles array telling whether we have subtitle this second
        for subtitle in subtitles:
            start = int(subtitle.start.total_seconds()) + offset
            end = int(subtitle.end.total_seconds()) + offset
            for i in range(start, end):
                subtitles_presence_array[i] = 1

        diff = length - len(subtitles_presence_array)
        if diff > 0:
            subtitles_presence_array = np.concatenate((subtitles_presence_array, np.zeros(diff)), axis=None)

        return np.concatenate((np.zeros(padding),
                               np.concatenate((subtitles_presence_array, np.zeros(padding)), axis=None)),
                               axis=None)

def redistribute(old_dist, size):
    num_ones = 0
    new_dist = []
    i = 0
    for val in old_dist:
        if val == 1:
            num_ones += 1
        if i == size:
            new_dist.append(int(num_ones >= (size / 2)))
            num_ones = 0
            i = 0
        i += 1
    return new_dist


# MFCC - plotting, to be used in get_speech_features
#plt.matshow(features_mfcc)
#plt.title('MFCC')

#use for plotting
#filterbank_features = filterbank_features.T

# identify hot areas, with high probabilities of speech
# step size = filterbank[0]
# do something using all rows instead..

#print(len(filterbank_features))
#print(X)
#plt.plot(X)
#plt.ylabel('speech probability')
#plt.show()

#plt.matshow(filterbank_features)
#plt.title('Filter bank')
#plt.show()
#matplotlib inline
