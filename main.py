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
    # used for training
    # "SF_ANYTIME_9259", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "ABMUI", "ABMRG"
    id = "SF_ANYTIME_9564"

    speech_features = get_speech_features("datasets/" + id + ".wav")
    clf = load('models/rfnext.joblib')

    start = time.time()
    predicted = clf.predict(speech_features)
    end = time.time()

    predicted_sec = per_sec(predicted, windows_per_sec=200)
    subtitles_array = get_subtitles_array("datasets/" + id + "_no.srt", length=len(predicted_sec))
    visualize_prediction_fraction(predicted, subtitles_array, 200)

    print("accuracy rfnext: ", get_accuracy(predicted_sec, subtitles_array))

    clf = load('models/rfsuper.joblib')

    start = time.time()
    predicted = clf.predict(speech_features)
    end = time.time()
    predicted_sec = per_sec(predicted, windows_per_sec=200)
    print("accuracy rfsuper: ", get_accuracy(predicted_sec, subtitles_array))
    #plot_roc(predictedDist, labelDist)
    '''
    print("time taken to predict: ", end - start, "s")
    for i in range(0, 4):
        padding = 100
        subtitles_array = get_subtitles_array("datasets/" + id + "_no.srt", length=len(predicted_sec), offset=i, padding=padding)

        chunk_start = padding
        chunk_end = chunk_start + len(predicted_sec)
        #visualize_prediction(predicted, subtitles_array[chunk_start:chunk_end])
        print("unsync at: " + str(i) + ", " + str(search_sync(predicted_sec, subtitles_array, chunk_start, chunk_end, padding)))
    '''

def search_sync(predicted_sec, subtitles, chunk_start, chunk_end, steps):
    # search forwards
    current_similarity = get_similarity(predicted_sec, subtitles[chunk_start:chunk_end])
    print("current_similarity ", current_similarity)

    unsync = False
    print("start searching forwards")
    i = 1
    while i < steps and chunk_end + i < len(subtitles):
        sim = get_similarity(predicted_sec, subtitles[chunk_start + i:chunk_end + i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            unsync = True
        i += 1

    print("start searching backwards")
    # search backwards
    i = 1
    while i < steps:
        sim = get_similarity(predicted_sec, subtitles[chunk_start - i:chunk_end - i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            unsync = True
        i += 1

    return unsync

def create_labels(subtitles_array, features_length):
    labels = []
    for presence in subtitles_array:
        labels = np.concatenate((labels, np.full((200), presence)), axis=None)

    if len(labels) < features_length:
        missing = features_length - len(labels)
        labels = np.concatenate((labels, np.zeros(missing)), axis=None)

    print("lengths...")
    print(features_length)
    print(len(labels))
    return labels[0:features_length]

def train():

    #X, y = get_features("SF_ANYTIME_9259", 100000, "chunk2")
    X = []
    y = []
    for id in ["ABMQU", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "NGPlus_163151_183450", "FoxInSeasonStacked_YRG902"]:
        # create chunks
        features = get_speech_features("datasets/" + id + ".wav")
        subtitles_array = get_subtitles_array("datasets/" + id + "_no.srt", int(len(features) / 200))
        labels = create_labels(subtitles_array, len(features))
        print("lengths...")
        print(len(features))
        print(len(labels))

        if len(X):
            X = np.vstack((X, features))
            y = np.hstack((y, labels))
        else:
            X = features
            y = labels

    clf = train_random_forest(X, y)

main()
