import matplotlib.pyplot as plt

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

def visualize_prediction_strict(predicted, labels):
    n1s = 0
    predictedDist = []
    i = 0
    for prediction in predicted:
        if prediction == 1:
            n1s += 1
        if i == 200:
            if n1s >= 100:
                predictedDist.append(200)
            else:
                predictedDist.append(0)
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
