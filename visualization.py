import matplotlib.pyplot as plt
from scipy import spatial

def visualize_prediction(predicted, labels):
    n1s = 0
    predictedDist = []
    i = 0
    for prediction in predicted:
        if prediction == 1:
            n1s += 1
        if i == 200:
            predictedDist.append(n1s)
            n1s = 0
            i = 0
        i += 1

    n1s = 0
    labelDist = []
    i = 0
    for label in labels:
        if label == 1:
            n1s += 1
        if i == 200:
            labelDist.append(n1s)
            n1s = 0
            i = 0
        i += 1

    # zeros threshold?
    plt.plot(predictedDist)
    plt.plot(labelDist)
    plt.ylabel('predicted')
    plt.show()


    cosine = 1 - spatial.distance.cosine(predicted, labels)
    print("cosine similarity is: ", cosine)
    jaccard = 1 - spatial.distance.jaccard(predicted, labels)
    print("jaccard similarity is: ", jaccard)
