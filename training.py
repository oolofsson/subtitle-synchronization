from preprocessing import *

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_svm(audio_filename, subtitles):
    clf = svm.SVC(gamma='scale', kernel='rbf', C=100, verbose=True)

    X = get_speech_features("chunks/" + audio)
    y = subtitles

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/svm.joblib')
    return clf
def train_random_forest(audio_filename, subtitles):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0, verbose=True)

    X = get_speech_features("chunks/" + audio)
    y = subtitles

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/rf.joblib')
    return clf

def train_mlp(audio_filename, subtitles):
    clf = MLPClassifier(hidden_layer_sizes=(30,30,30))

    X = get_speech_features("chunks/" + audio_filename)
    y = subtitles

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/mlp.joblib')
    return clf
