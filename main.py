from training import *
from preprocessing import *
from visualization import *
from evaluation import *
import wave
import contextlib
import time



def main():
    predict()

def predict():
    which_chunk = 2
    filename = "FoxPlus_YSS104"

    features, labels = get_features(filename, 500000, "chunk" + str(which_chunk))

    clf = load('models/rf.joblib')

    start = time.time()
    predicted = clf.predict(features)
    end = time.time()

    predicted = per_sec(predicted, windows_per_sec=200)
    labels = per_sec(labels, windows_per_sec=200)

    #plot_roc(predictedDist, labelDist)
    print("time taken to predict: ", end - start, "s")
    for i in range(0, 4):
        subtitles_array = get_subtitles_array("datasets/" + filename + "_no.srt", offset=i)
        chunk_start = which_chunk * len(predicted)
        chunk_end = chunk_start + len(predicted)
        print("unsync at: " + str(i) + ", " + str(search_sync(predicted, subtitles_array, chunk_start, chunk_end)))


def search_sync(predicted, subtitles, chunk_start, chunk_end):
    # search forwards
    current_similarity = get_similarity(predicted, subtitles[chunk_start:chunk_end])
    print("current_similarity ", current_similarity)
    # subtitles = add_offset(subtitles), start and end...
    # if chunk_start + predicted > len(subtitles), add more offset to end
    # chunk_start = chunk_start + offset
    # chunk_end = chunk_end + offset
    i = 2
    while i < 100:
        sim = get_similarity(predicted, subtitles[chunk_start + i:chunk_end + i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1

    print("start searching backwards")
    # search backwards
    i = 2
    while i < 100:
        sim = get_similarity(predicted, subtitles[chunk_start - i:chunk_end - i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1

    return False


def get_features(id, chunk_length_ms, chunk):
    subtitles_array = get_subtitles_array("datasets/" + id + "_no.srt")
    create_audio_chunks("datasets/" + id + ".wav", chunk_length_ms) # stored in chunks folder
    chunk_subtitles = get_chunk_subtitles(subtitles_array) # based on chunks

    return get_speech_features("chunks/" + chunk + ".wav"), chunk_subtitles[chunk + ".wav"]

def train():

    #X, y = get_features("SF_ANYTIME_9259", 100000, "chunk2")
    X = []
    y = []
    for id in ["SF_ANYTIME_9259", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "ABMUI", "ABMRG"]:
        Xadd, yadd = get_features(id, 100000, "chunk1")
        if len(X):
            X = np.vstack((X, Xadd))
            y = np.hstack((y, yadd))
        else:
            X = Xadd
            y = yadd

    clf = train_random_forest(X, y)

main()
