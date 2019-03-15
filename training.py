import preprocessing

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_svm():
    clf = svm.SVC(gamma='scale', kernel='rbf', C=100, verbose=True)

    subtitles = enerate_subtitled("AAFPU_no.srt")
    X = get_speech_features("chunks/chunk0.wav")
    y = subtitles["chunk0"]

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'random.joblib')
    return clf
def train_random_forest():
    clf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0, verbose=True)

    subtitles = generate_subtitled("AAFPU_no.srt")
    X = get_speech_features("chunks/chunk0.wav")
    y = subtitles["chunk0"]

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'random.joblib')
    return clf

def train_mlp():
    clf = MLPClassifier(hidden_layer_sizes=(30,30,30))

    subtitles = generate_subtitled("AAFPU_no.srt")
    X = get_speech_features("chunks/chunk0.wav")
    y = subtitles["chunk0"]

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'random.joblib')
    return clf
