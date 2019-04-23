from training import *
from preprocessing import *
from visualization import *
from synchronization import *
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
import ffmpeg

s3_resource = boto3.resource('s3',
    aws_access_key_id=config.aws['ACCESS_KEY'],
    aws_secret_access_key=config.aws['SECRET_KEY']
)

def main():
    train()


def test_sync():
    id = "NGPlus_163151_183450"

    speech_features = get_speech_features("datasets/" + id + ".wav")
    clf = load('models/rfnext.joblib')

    predicted = clf.predict(speech_features)
    padding = 100
    predicted_summarized = redistribute(predicted, 200)
    start = padding
    end = start + len(predicted_summarized)
    subtitles_presence_array = get_subtitles_presence_array("datasets/" + id + "_no_shift2.srt", length=len(predicted_summarized), padding=padding)

    synchronize(predicted_summarized, subtitles_presence_array, start, end, padding)
# -------------------------------- Get Unsync Search --------------------------------

def clean_detection_files():
    if os.path.exists('audio.wav'):
        os.remove('audio.wav')
    if os.path.exists('subtitles.srt'):
        os.remove('subtitles.srt')
    if os.path.exists('subtitles.vtt'):
        os.remove('subtitles.vtt')
    if os.path.exists('audio.mp4'):
        os.remove('audio.mp4')

def convert():
    try:
        ffmpeg.input('audio.mp4').output('audio.wav', format='wav', ar='16k', loglevel='panic').run()
        ffmpeg.input('subtitles.vtt').output('subtitles.srt', format='srt', loglevel='panic').run()
    except:
        return False
    return True

def run_unsync_search():
    clean_detection_files()

    bucket = s3_resource.Bucket('get-internal-import')
    processed_ids = defaultdict(lambda:False)

    clf = load('models/rfnext.joblib')

    for object in bucket.objects.all():
        keys = object.key.split('/')
        #print(object.key)
        source = keys[0]
        id = keys[1]

        if processed_ids[id]:
            continue

        try:
            s3_resource.Object('get-internal-import', source + '/' + id + '/aac_128000_48000_AACChannelLayout.CL_STEREO_v2.mp4').download_file('audio.mp4')
        except:
            print("Could not download file: " + source + '/' + id + '/aac_128000_48000_AACChannelLayout.CL_STEREO_v2.mp4')
            processed_ids[id] = True
            continue

        r = requests.get('https://subtitles.getvideo.cloud/' + id + '_no.vtt')
        if len(r.content) < 1000:
            print("subtitles not found for source: ", source, ". With id: ", id)
        else:
            open('subtitles.vtt', 'wb').write(r.content)

            if convert(): # mp4 -> vaw, vtt -> srt
            # file pair ready for sync test
                process(source, id, clf)
            else:
                print("Could not convert " + id)

        clean_detection_files()
        processed_ids[id] = True

def process(source, id, clf):
    print("Processing " + source + "/" + id)
    print("Extracting speech features.")
    speech_features = get_speech_features("audio.wav")
    predicted = clf.predict(speech_features)
    predicted_summarized = redistribute(predicted, size=200)

    padding = 100
    subtitles_presence_array = get_subtitles_presence_array("subtitles.srt", length=len(predicted_summarized), padding=padding, offset=0)

    start = padding
    end = start + len(predicted_summarized)
    if is_unsynchronized(predicted_summarized, subtitles_presence_array, start, end, padding):
        print("Detected Unsynchronization in source: ", source, ". id: ", id)
    else:
        print("Detected Synchronization in source: ", source, ". id: ", id)

# -------------------------------- Evaluation --------------------------------

def evaluate():
    # used for training default
    # "SF_ANYTIME_9259", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "ABMUI", "ABMRG"
    # "ABMQU", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202", "NGPlus_163151_183450", "FoxInSeasonStacked_YRG902"
    id = "CMRE0000000001081770"

    speech_features = get_speech_features("datasets/" + id + ".wav")

    clf = load('models/rfnext.joblib')

    start = time.time()
    predicted = clf.predict(speech_features)
    end = time.time()
    print("time taken to predict: ", end - start, "s")

    predicted_summarized = redistribute(predicted, 200)
    subtitles_presence_array = get_subtitles_presence_array("datasets/" + id + "_no.srt", length=len(predicted_summarized))
    print("accuracy rf: ", get_accuracy(predicted_summarized, subtitles_presence_array))
    print(subtitles_presence_array)
    #visualize_prediction(predicted_sec, subtitles_array_sec)
    #plot_roc(predictedDist, labelDist)

    for i in range(0, 4):
        padding = 100
        subtitles_presence_array = get_subtitles_presence_array("datasets/" + id + "_no.srt", length=len(predicted_summarized), offset=i, padding=padding)

        start = padding
        end = start + len(predicted_summarized)
        #visualize_prediction(predicted, subtitles_array[start:end])
        print("unsync at: " + str(i) + ", " + str(is_unsynchronized(predicted_summarized, subtitles_presence_array, start, end, padding)))

main()
