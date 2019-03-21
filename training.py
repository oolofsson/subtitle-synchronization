from preprocessing import *

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop

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

def model_lstm(input_shape):
    inp = Input(shape=input_shape)
    model = inp

    if input_shape[0] > 2: model = Conv1D(filters=24, kernel_size=(3), activation='relu')(model)
#    if input_shape[0] > 0: model = TimeDistributed(Conv1D(filters=24, kernel_size=3, activation='relu'))(model)
    model = LSTM(16)(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(16)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    model = Model(inp, model)
    return model

#reccurentnet
