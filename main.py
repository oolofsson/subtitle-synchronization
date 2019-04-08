from training import *
from preprocessing import *
from visualization import *
from evaluation import *
from collections import defaultdict
import wave
import contextlib
import time
import boto3
import os
import requests
import argparse
import config

s3_resource = boto3.resource('s3',
    aws_access_key_id=config.aws['ACCESS_KEY'],
    aws_secret_access_key=config.aws['SECRET_KEY']
)

def main():
    run_detection()

def convert(id):
    os.system('ffmpeg -i audio.mp4 -ar 16000 ' + id + '.wav')
    os.system('rm audio.mp4')
    os.system('ffmpeg -i subtitles.vtt ' + id + '.srt')
    os.system('rm subtitles.vtt')


def run_detection():
    bucket = s3_resource.Bucket('get-internal-import')
    processed_ids = defaultdict(lambda:False)
    clf = load('models/rf.joblib')

    for object in bucket.objects.all():
        keys = object.key.split('/')
        print("this is")
        print(object.key)
        source = keys[0]
        id = keys[1]
        if processed_ids[id]:
            continue

        try:
            s3_resource.Object('get-internal-import', source + '/' + id + '/aac_128000_48000_AACChannelLayout.CL_STEREO_v2.mp4').download_file('audio.mp4')
        except:
            print("Could not download file: " + source + '/' + id + '/aac_128000_48000_AACChannelLayout.CL_STEREO_v2.mp4')
            continue

        r = requests.get('https://subtitles.getvideo.cloud/' + id + '_no.vtt')
        if len(r.content):
            open('subtitles.vtt', 'wb').write(r.content)
            convert(id) # mp4 -> vaw, vtt -> srt

            speech_features = get_speech_features(id + ".wav")
            predicted = clf.predict(speech_features)
            predicted_sec = per_sec(predicted, windows_per_sec=200)

            padding = 100
            subtitles_array_sec = get_subtitles_array(id + ".srt", length=len(predicted_sec), padding=padding)

            chunk_start = padding
            chunk_end = chunk_start + len(predicted_sec)
            if is_unsynchronized(predicted_sec, subtitles_array_sec, chunk_start, chunk_end, padding):
                print("Detected unsynchronization in source: ", source, ". id: ", id)

            processed_ids[id] = True

            os.system('rm ' + keys[1] + '.mp4')
            os.system('rm ' + keys[1] + '.srt')
        else:
            print("subtitles not found for source: ", source, ". id: ", id)

def evaluate():
    # used for training
    # "SF_ANYTIME_9259", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "ABMUI", "ABMRG"
    id = "SF_ANYTIME_9564"

    speech_features = get_speech_features("datasets/" + id + ".wav")

    clf = load('models/rf.joblib')

    start = time.time()
    predicted = clf.predict(speech_features)
    end = time.time()
    #print("time taken to predict: ", end - start, "s")

    predicted_sec = per_sec(predicted, windows_per_sec=200)
    subtitles_array_sec = get_subtitles_array("datasets/" + id + "_no.srt", length=len(predicted_sec))
    print("accuracy rf: ", get_accuracy(predicted_sec, subtitles_array))

    visualize_prediction(predicted_sec, subtitles_array_sec)
    #plot_roc(predictedDist, labelDist)
    '''
    for i in range(0, 4):
        padding = 100
        subtitles_array = get_subtitles_array("datasets/" + id + "_no.srt", length=len(predicted_sec), offset=i, padding=padding)

        chunk_start = padding
        chunk_end = chunk_start + len(predicted_sec)
        #visualize_prediction(predicted, subtitles_array[chunk_start:chunk_end])
        print("unsync at: " + str(i) + ", " + str(search_sync(predicted_sec, subtitles_array, chunk_start, chunk_end, padding)))
    '''

def is_unsynchronized(predicted_sec, subtitles_array_sec, chunk_start, chunk_end, steps):
    current_similarity = get_similarity(predicted_sec, subtitles[chunk_start:chunk_end])
    # search forwards
    i = 1
    while i < steps and chunk_end + i < len(subtitles_array_sec):
        sim = get_similarity(predicted_sec, subtitles_array_sec[chunk_start + i:chunk_end + i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1

    # search backwards
    i = 1
    while i < steps:
        sim = get_similarity(predicted_sec, subtitles_array_sec[chunk_start - i:chunk_end - i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1
    return False

def create_labels_from_subtitles(subtitles_array_sec, features_length):
    labels = []
    for presence in subtitles_array_sec:
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
        subtitles_array_sec = get_subtitles_array("datasets/" + id + "_no.srt", int(len(features) / 200))
        labels = create_labels(subtitles_array_sec, len(features))
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
