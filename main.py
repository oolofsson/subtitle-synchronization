from training import *
from preprocessing import *

def main():
    subtitles = generate_subtitles("datasets/AAFPU_no.srt")
    split_audio("datasets/AAFPU.wav", chunk_length_ms=600000) # stored in chunks folder
    split_subtitles(subtitles)


    #clf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0, verbose=True)
    #clf = MLPClassifier(hidden_layer_sizes=(30,30,30))
    #clf = svm.SVC(gamma='scale', kernel='rbf', C=100, verbose=True)

main()
