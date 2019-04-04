import srt
import shutil
import numpy as np
import datetime
import os
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from pydub import AudioSegment
from pydub.utils import make_chunks
from joblib import dump, load
from sklearn.preprocessing import scale

def create_audio_chunks(audio_file, chunk_length_ms):
    myaudio = AudioSegment.from_file(audio_file , "wav")
    chunks = make_chunks(myaudio, chunk_length_ms) # Make chunks of one sec

    # remove earlier chunks

    shutil.rmtree("chunks", ignore_errors=True)
    os.makedirs("chunks")

    # Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = "chunks/chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

def get_chunk_subtitles(subtitles):
    chunks = os.listdir('chunks')

    chunk_size = len(get_speech_features('chunks/' + sorted(chunks)[0]))
    last_chunk_size = len(get_speech_features('chunks/' + sorted(chunks)[-1]))
    chunk_subtitles = {}
    first = 0
    last = int(chunk_size / 200)
    for chunk in sorted(chunks):
        chunk_subtitles[chunk] = subtitles[first:last]
        first = last + 1
        last += chunk_size + 1

    chunk_subtitles[sorted(chunks)[-1]] = chunk_subtitles[sorted(chunks)[-1]][0:last_chunk_size]
    return chunk_subtitles

def get_speech_features(audio_file):
    frequency_sampling, audio_signal = wavfile.read(audio_file)
    # audio_signal = audio_signal[:1500000]

    features_mfcc = mfcc(signal=audio_signal, samplerate=frequency_sampling, winlen=0.025, nfft=512)

    print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    print('Length of each feature =', features_mfcc.shape[1])

    features_mfcc = features_mfcc.T
    #plt.matshow(features_mfcc)
    #plt.title('MFCC')
    print("frequency is: ", frequency_sampling)
    filterbank_features = logfbank(signal=audio_signal, samplerate=frequency_sampling, winlen=0.025, nfft=512)

    print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
    print('Length of each feature =', filterbank_features.shape[1])

    return scale(np.array(filterbank_features), axis=0, with_mean=True, with_std=True, copy=True)

def get_subtitles_array(srt_file, offset=0):
    with open(srt_file) as file:
        srt_string = file.read()
        subtitles = list(srt.parse(srt_string))
        subtitles_end = subtitles[-1].end.total_seconds()
        subtitles_array = np.zeros(int(subtitles_end) + offset)
        # generate subtitles array telling whether we have subtitle this second
        for subtitle in subtitles:
            start = int(subtitle.start.total_seconds()) + offset
            end = int(subtitle.end.total_seconds()) + offset
            for i in range(start, end):
                subtitles_array[i] = 1
        return subtitles_array

def per_sec(array, windows_per_sec):
    n1s = 0
    dist = []
    i = 0
    for element in array:
        if element == 1:
            n1s += 1
        if i == windows_per_sec:
            if n1s >= windows_per_sec / 2:
                dist.append(1)
            else:
                dist.append(0)
            n1s = 0
            i = 0
        i += 1

    return dist


# MFCC - plotting, to be used in get_speech_features

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
