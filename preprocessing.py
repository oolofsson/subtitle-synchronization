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

def split_audio(audio_file, chunk_length_ms):
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

def split_subtitles(subtitles):
    chunks = os.listdir('chunks')

    chunk_size = len(get_speech_features('chunks/' + sorted(chunks)[0]))
    last_chunk_size = len(get_speech_features('chunks/' + sorted(chunks)[-1]))
    splitted_subtitles = {}
    first = 0
    last = chunk_size
    for chunk in sorted(chunks):
        splitted_subtitles[chunk] = subtitles[first:last]
        first = last + 1
        last += chunk_size + 1

    splitted_subtitles[sorted(chunks)[-1]] = splitted_subtitles[sorted(chunks)[-1]][0:last_chunk_size]

    return splitted_subtitles



def get_speech_features(audio_file):
    frequency_sampling, audio_signal = wavfile.read(audio_file)
    # audio_signal = audio_signal[:1500000]

    features_mfcc = mfcc(signal=audio_signal, samplerate=frequency_sampling, nfft=1200)

    print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    print('Length of each feature =', features_mfcc.shape[1])

    features_mfcc = features_mfcc.T
    #plt.matshow(features_mfcc)
    #plt.title('MFCC')

    filterbank_features = logfbank(signal=audio_signal, samplerate=frequency_sampling, nfft=1200)

    print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
    print('Length of each feature =', filterbank_features.shape[1])

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
    return scale(np.array(filterbank_features), axis=0, with_mean=True, with_std=True, copy=True)

def generate_subtitles(srt_file):
    with open(srt_file) as file:
        srt_string = file.read()
        subtitle_generator = srt.parse(srt_string)
        # generate subtitles array telling whether we have subtitle this second
        subtitles = np.zeros(100000000)
        for subtitle in list(subtitle_generator):
            start = int(subtitle.start.total_seconds())
            end = int(subtitle.end.total_seconds())
            for i in range(start*200, end*200):
                subtitles[i] = 1
        return subtitles
