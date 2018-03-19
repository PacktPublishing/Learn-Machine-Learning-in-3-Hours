import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, 0]

    # Plot raw data.
    fig, ax = plt.subplots()
    ax.set_xlabel('Stress 1 (MPa)')
    ax.set_ylabel('Stress 2 (MPa)')
    sc = ax.scatter(X, y)

    # Check for error values and remove.
    data = data[np.all(data > 0, axis=1)]

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, 0]
    X = X[:, np.newaxis]

    # Scale data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Plot scaled data.
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Stress 1 (Scaled)')
    ax2.set_ylabel('Stress 2 (MPa)')
    sc = ax2.scatter(X_scaled, y)

    plt.show()

    # Generate training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.5)



