from training import *
from preprocessing import *
from visualization import *

def main():
    subtitles = generate_subtitles("datasets/AAFPU_no.srt")
    split_audio("datasets/AAFPU.wav", chunk_length_ms=600000) # stored in chunks folder
    splitted_subtitles = split_subtitles(subtitles)

    #clf = train_mlp("chunk0.wav", splitted_subtitles["chunk0.wav"])

    clf = load('models/mlp.joblib')
    predicted = clf.predict(get_speech_features("chunks/chunk1.wav"))

    visualize_prediction(predicted, splitted_subtitles["chunk0.wav"])



main()
