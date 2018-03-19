import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


def plot_legend(ax_in):

    """Add chicken and duck legend to scatter plot"""

    purple_patch = mpatches.Patch(color='purple', label='Chicken')
    yellow_patch = mpatches.Patch(color='yellow', label='Duck')
    ax_in.legend(handles=[purple_patch, yellow_patch])


if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    X = np.load(FILENAME)

    # Check for nans and remove.
    X = X[~np.isnan(X).any(axis=1)]

    # Split into input data and labels.
    y = X[:, -1]
    X = X[:, :-1]

    # Plot raw data.
    fig, ax = plt.subplots()
    ax.set_xlabel('Egg Length (cm)')
    ax.set_ylabel('Egg Weight (g)')
    sc = ax.scatter(X[:, 0], X[:, 1], c=y)
    plot_legend(ax)

    # Scale data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Plot scaled data.
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Egg length scaled')
    ax2.set_ylabel('Egg weight scaled')
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)

    plot_legend(ax2)

    plt.show()

