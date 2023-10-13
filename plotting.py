import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(y_data, labels):
    class_counts = np.bincount(y_data)
    plt.bar(labels, class_counts)
    plt.show()
    print(class_counts)

def plot_graph_one(x_data, fs):
    time_axis = np.arange(0, len(x_data)) / fs
    plt.plot(time_axis, x_data)
    plt.show()

def plot_graph_two(x_data, y_data, fs):
    time_axis = np.arange(0, len(x_data)) / fs
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(time_axis, x_data)
    ax2.plot(time_axis, y_data)
    plt.show()

