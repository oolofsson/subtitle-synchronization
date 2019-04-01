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
    which_chunk = 1
    X, y = get_features("NGPlus_170158_268038", 800000, "chunk" + str(which_chunk))
    subtitles = generate_subtitle_array("datasets/NGPlus_170158_268038_no.srt")

    clf = load('models/rf.joblib')

    start = time.time()
    predicted = clf.predict(X)
    end = time.time()
    print("time taken to predict: ", end - start, "s")

    predictedDist = generate_dist(predicted)
    labelDist = generate_dist(y)
    subtitleDist = generate_dist(subtitles)

    #visualize_prediction(predictedDist, labelDist)

    print("accuracy:")
    print(accuracy(predictedDist, labelDist))
    #print("start shifting:")
    chunk_start = which_chunk * len(predictedDist)
    chunk_end = chunk_start + len(predictedDist)
    search_sync(predictedDist, subtitleDist, chunk_start, chunk_end)

def search_sync(predictedDist, subtitleDist, chunk_start, chunk_end):
    # search forwards
    i = 0
    print("start searching forwards")
    print("predictedDist Length: ")
    print(len(predictedDist))
    print("subtitleDist Length: ")
    print(len(subtitleDist[chunk_start + i:chunk_end + i]))
    while i < 20:
        print(similarity(predictedDist, subtitleDist[chunk_start + i:chunk_end + i]))
        i += 2

    print("start searching backwards")
    # search backwards
    i = 0
    while i < 20:
        print(similarity(predictedDist, subtitleDist[chunk_start - i:chunk_end - i]))
        i += 2

def get_features(id, chunk_length_ms, chunk):
    subtitle_array = generate_subtitle_array("datasets/" + id + "_no.srt")
    create_audio_chunks("datasets/" + id + ".wav", chunk_length_ms) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitle_array) # based on chunks

    return get_speech_features("chunks/" + chunk + ".wav"), splitted_subtitles[chunk + ".wav"]

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
