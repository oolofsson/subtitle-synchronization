from preprocessing import *

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def train_svm(X, y):

    param_grid = {'C': [1,10,100,1000],
                  'gamma': [1,0.1,0.001,0.0001],
                  'kernel': ['linear','rbf']}

    #clf = GridSearchCV(svm.SVC(),param_grid, refit = True, verbose=2)
    clf = svm.SVC(kernel='rbf', gamma='scale', C=100, verbose=True)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/svm.joblib')
    return clf
def train_random_forest(X, y):
    param_grid = {'max_depth': [3, None],
                  'max_features': [1, 3, 10],
                  'min_samples_split': [2, 3, 10],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    clf = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid, refit = True, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/rf.joblib')
    return clf

def train_mlp(X, y):

    param_grid = {'solver': ['lbfgs', 'adam'],
                  'max_iter': [2000],
                  'alpha': 10.0 ** -np.arange(1, 4),
                  'hidden_layer_sizes': [10,15],
                  'random_state':[0,5,9]}
    clf = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/mlp.joblib')
    return clf

def train_rnn(X, y):
    print(X.shape)
    #X = X.reshape(X.shape[1], X.shape[0], X.shape[1])
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    #print(X.shape)
    model = Sequential()
    model.add(LSTM(128, input_shape=X.shape[1:], activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))


    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='softmax'))
    model.add(Dropout(0.2))

    model.compile(loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3, decay=1e-5),
        metrics=['accuracy'])

    model.fit(X, y, epochs=3)
    dump(model, 'models/rnn.joblib')

    return model


def train_cnn_lstm(X, y):

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    model = Sequential()
    model.add(Conv1D(filters=26, kernel_size=(1), activation='relu'))
    model.add(LSTM(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    model.fit(X, y, epochs=3)
    dump(model, 'models/cnnlstm.joblib')
    return model
