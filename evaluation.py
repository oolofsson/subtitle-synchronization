from scipy import spatial
from sklearn import metrics
import matplotlib.pyplot as plt

def get_similarity(predicted, labels):
    return 1 - spatial.distance.yule(labels, predicted)

def get_accuracy(predicted, labels):
    return metrics.accuracy_score(labels, predicted)


def plot_roc(predicted, labels):
    fpr, tpr, thresholds = metrics.roc_curve(predicted, labels)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
