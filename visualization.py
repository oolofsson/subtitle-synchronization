import matplotlib.pyplot as plt

def visualize_prediction(predicted, labels):
    # zeros threshold?
    plt.plot(predicted)
    plt.plot(labels)
    plt.ylabel('predicted vs actual')
    plt.show()
