from preprocessing import *

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def create_labels_from_subtitles(subtitles_presence_array, features_length):
    labels = []
    for presence in subtitles_presence_array:
        labels = np.concatenate((labels, np.full((200), presence)), axis=None)

    if len(labels) < features_length:
        missing = features_length - len(labels)
        labels = np.concatenate((labels, np.zeros(missing)), axis=None)

    return labels[0:features_length]

def train():

    #X, y = get_features("SF_ANYTIME_9259", 100000, "chunk2")
    X = []
    y = []
    for id in ["ABMUI"]: #, "ABMQU", "AAFPU", "ABMQU", "SF_ANYTIME_9547", "CMRE0000000001000202"]:

        # create chunks
        features = get_speech_features("datasets/" + id + ".wav")
        subtitles_presence_array = get_subtitles_presence_array("datasets/" + id + "_no.srt", int(len(features) / 200))
        labels = create_labels_from_subtitles(subtitles_presence_array, len(features))
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

def train_svm(X, y):

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['rbf']}

    clf = RandomizedSearchCV(svm.SVC(), param_grid, refit = True, n_jobs=-1, verbose=2)
    #clf = svm.SVC(kernel='rbf', gamma='scale', C=100, verbose=True)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/svm_one.joblib')
    return clf

def train_random_forest(X, y):
    param_grid = {'max_depth': [3, None],
                  'max_features': [1, 3, 10],
                  'min_samples_split': [2, 3, 10],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    clf = RandomizedSearchCV(RandomForestClassifier(n_estimators=20), param_grid, n_jobs=-1, refit = True, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/rf_one.joblib')
    return clf

def train_mlp(X, y):
    param_grid = {'solver': ['lbfgs', 'adam'],
                  'max_iter': [1000,1100,1200,1300,
                  1400,1500,1600,1700,1800,1900,2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15),
                  'random_state': [0,1,2,3,4,5,6,7,8,9]}
    clf = RandomizedSearchCV(MLPClassifier(), param_grid, n_jobs=-1, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/mlp_one.joblib')
    return clf
