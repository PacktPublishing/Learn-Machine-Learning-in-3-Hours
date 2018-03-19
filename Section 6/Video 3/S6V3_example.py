import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, :-1]

    # Plot raw data.
    fig, ax = plt.subplots()
    ax.set_xlabel('Price Change 1')
    ax.set_ylabel('Price Change 2')
    sc = ax.scatter(X[:, 0], X[:, 1], c=y)

    # Check for error values and remove
    # ???

    # Scale data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Plot scaled data.
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Price Change 1 (Scaled)')
    ax2.set_ylabel('Price Change 2 (Scaled)')
    sc = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)

    plt.show()

    # Generate training and testing set.
    mid_point = int(len(X_scaled) / 2)

    X_train = X_scaled[:mid_point, :]
    X_test = X_scaled[mid_point:, :]
    y_train = y[:mid_point]
    y_test = y[mid_point:]
