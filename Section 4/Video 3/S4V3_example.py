import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


def plot_legend(ax_in):

    """Add legend to scatter plot"""

    purple_patch = mpatches.Patch(color='purple', label='Labour')
    yellow_patch = mpatches.Patch(color='yellow', label='Conservative')
    green_patch = mpatches.Patch(color='green', label='Liberal Democrat')
    ax_in.legend(handles=[purple_patch, yellow_patch, green_patch])


if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, :-1]

    # Plot raw data.
    fig, ax = plt.subplots()
    ax.set_xlabel('Left - Right leaning')
    ax.set_ylabel('Increased military spending favorability')
    sc = ax.scatter(X[:, 0], X[:, 1], c=y)
    plot_legend(ax)

    # Check for error values and remove.
    data = data[np.all(data > -1, axis=1)]

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, :-1]

    # Scale data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Plot scaled data.
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('L-R leaning scaled')
    ax2.set_ylabel('Increase military spending favorability scaled')
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)

    plot_legend(ax2)

    # Generate training and testing set.
    mid_point = int(len(X_scaled) / 2)

    X_train = X_scaled[:mid_point, :]
    X_test = X_scaled[mid_point:, :]
    y_train = y[:mid_point]
    y_test = y[mid_point:]

    plt.show()









