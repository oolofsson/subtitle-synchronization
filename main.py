from training import *
from preprocessing import *
from visualization import *
from evaluation import *
import wave
import contextlib

def main():
    train()

def predict():
    subtitles = generate_subtitles("datasets/ABMQU_no.srt")
    create_audio_chunks("datasets/ABMQU.wav", chunk_length_ms=500000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    clf = load('models/mlp.joblib')
    predicted = clf.predict(get_speech_features("chunks/chunk3.wav"))

    labels = splitted_subtitles["chunk3.wav"]
    visualize_prediction_strict(predicted, labels)
    print("similarity:")
    print(similarity(predicted, labels))
    print("accuracy:")
    print(accuracy(predicted, labels))

def train():
    subtitles = generate_subtitles("datasets/SF_ANYTIME_9259_no.srt")
    create_audio_chunks("datasets/SF_ANYTIME_9259.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    X = get_speech_features("chunks/chunk1.wav")
    y = splitted_subtitles["chunk1.wav"]

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
    create_audio_chunks("datasets/CMRE0000000001000202.wav", chunk_length_ms=100000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles) # based on chunks

    X = np.vstack((X, get_speech_features("chunks/chunk0.wav")))
    y = np.hstack((y, splitted_subtitles["chunk0.wav"]))

    clf = train_svm(X, y)

main()
