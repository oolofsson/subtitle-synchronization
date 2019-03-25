from training import *
from preprocessing import *
from visualization import *
from evaluation import *
import wave
import contextlib

def main():
    predict()

def predict():
    subtitles = generate_subtitles("datasets/AAFPU_no.srt")
    create_audio_chunks("datasets/AAFPU.wav", chunk_length_ms=500000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    clf = load('models/mlp.joblib')
    X = get_speech_features("chunks/chunk2.wav")

    #X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    #predicted = clf.predict(X).ravel()

    predicted = clf.predict(X)

    labels = splitted_subtitles["chunk2.wav"]

    predictedDist, labelDist = generate_dist(predicted, labels)
    visualize_prediction(predictedDist, labelDist)

    print(predictedDist)
    print(labelDist)
    print("similarity:")
    print(similarity(predictedDist, labelDist))
    print("accuracy:")
    print(accuracy(predictedDist, labelDist))
    print("start shifting:")
    shift(predictedDist, labelDist)

def shift(predictedDist, labelDist):
    for i in range(0, len(labelDist), 5):
        print(similarity(predictedDist, np.roll(labelDist, i)))
def train():
    subtitles = generate_subtitles("datasets/SF_ANYTIME_9259_no.srt")
    create_audio_chunks("datasets/SF_ANYTIME_9259.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    X = get_speech_features("chunks/chunk2.wav")
    y = splitted_subtitles["chunk2.wav"]

    subtitles = generate_subtitles("datasets/AAFPU_no.srt")
    create_audio_chunks("datasets/AAFPU.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles)

    X = np.vstack((X, get_speech_features("chunks/chunk1.wav")))
    y = np.hstack((y, splitted_subtitles["chunk1.wav"]))

    subtitles = generate_subtitles("datasets/ABMQU_no.srt")
    create_audio_chunks("datasets/ABMQU.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles)

    X = np.vstack((X, get_speech_features("chunks/chunk2.wav")))
    y = np.hstack((y, splitted_subtitles["chunk2.wav"]))


    subtitles = generate_subtitles("datasets/SF_ANYTIME_9547_no.srt")
    create_audio_chunks("datasets/SF_ANYTIME_9547.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    X = np.vstack((X, get_speech_features("chunks/chunk1.wav")))
    y = np.hstack((y, splitted_subtitles["chunk1.wav"]))

    subtitles = generate_subtitles("datasets/CMRE0000000001000202_no.srt")
    create_audio_chunks("datasets/CMRE0000000001000202.wav", chunk_length_ms=300000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    X = np.vstack((X, get_speech_features("chunks/chunk1.wav")))
    y = np.hstack((y, splitted_subtitles["chunk1.wav"]))

    clf = train_cnn_lstm(X, y)


main()
