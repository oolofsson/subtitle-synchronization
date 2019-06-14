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
    for id in ["ABMUI", "ABMQU"]: #, "AAFPU", "FoxPlus_YSS104", "CMRE0000000001000202"]:

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

    clf = train_svm(X, y)

def train_svm(X, y):

    param_grid = {'C': [0.001, 0.01, 0.1, 1],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['rbf', 'linear']}

    #clf = RandomizedSearchCV(svm.SVC(), param_grid, refit = True, n_jobs=-1, verbose=2)
    clf = svm.SVC(kernel='rbf', gamma=1, C=1, verbose=True)

    print("training started.")
    clf.fit(X, y)

    print("training finished.")
    print("best params")
    #print(clf.best_params_)
    dump(clf, 'models/svm_mfcc.joblib')
    return clf

def train_rf(X, y):
    '''
    param_grid = {'max_depth': [3, None],
                  'max_features': [1, 3, 10],
                  'min_samples_split': [2, 3, 10],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}'''

    param_grid = {'max_depth': [None],
                  'max_features': [10],
                  'min_samples_split': [3],
                  'bootstrap': [False],
                  'criterion': ['entropy']}

    clf = RandomizedSearchCV(RandomForestClassifier(n_estimators=20), param_grid, n_jobs=-1, refit = True, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")
    print("best params")
    print(clf.best_params_)
    dump(clf, 'models/rf_mfcc.joblib')
    return clf

def train_mlp(X, y):
    '''
    param_grid = {'solver': ['lbfgs', 'adam'],
                  'max_iter': [1000,1100,1200,1300,
                  1400,1500,1600,1700,1800,1900,2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15),
                  'random_state': [0,1,2,3,4,5,6,7,8,9]}
                  '''
    param_grid = {'solver': ['adam'],
                  'max_iter': [2000],
                  'alpha': [0.0000001],
                  'hidden_layer_sizes': [12],
                  'random_state': [9]}
    clf = RandomizedSearchCV(MLPClassifier(), param_grid, n_jobs=-1, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")
    print("best params")
    print(clf.best_params_)
    dump(clf, 'models/mlp_mfcc.joblib')
    return clf
