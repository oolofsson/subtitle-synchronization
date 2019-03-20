from preprocessing import *

from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_svm(X, y):

    param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf', 'poly']}

    clf = GridSearchCV(svm.SVC(),param_grid,refit = True, verbose=2)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/svm.joblib')
    return clf
def train_random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0, verbose=True)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/rf.joblib')
    return clf

def train_mlp(X, y):

    param_grid = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    clf = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1)

    print("training started.")
    clf.fit(X, y)
    print("training finished.")

    dump(clf, 'models/mlp.joblib')
    return clf


#convnet
def train_neural_network(X, y):
    pass

#reccurentnet
