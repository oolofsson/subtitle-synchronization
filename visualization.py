import matplotlib.pyplot as plt

def visualize_prediction(predicted, labels):
    # zeros threshold?
    plt.plot(predicted)
    plt.plot(labels)
    plt.ylabel('predicted vs actual')
    plt.show()

def visualize_prediction_fraction(array, labels, windows_per_sec):
    n1s = 0
    dist = []
    i = 0
    for element in array:
        if element == 1:
            n1s += 1
        if i == windows_per_sec:
            dist.append(n1s / windows_per_sec)
            n1s = 0
            i = 0
        i += 1
        
    plt.plot(dist)
    plt.plot(labels)
    plt.ylabel('percentage vs actual')
    plt.show()
