from scipy import spatial
from sklearn import metrics

def similarity(predicted, labels):
    return 1 - spatial.distance.cosine(labels, predicted)

def accuracy(predicted, labels):
    return metrics.accuracy_score(labels, predicted)
