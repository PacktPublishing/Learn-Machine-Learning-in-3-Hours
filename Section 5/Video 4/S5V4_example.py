import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#New library
from sklearn.svm import SVR

if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

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

    # Generate training and testing set.
    X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.5, random_state=42)

    # Above is code from previous video

    # Fit SVR model.
    svr = SVR(C=1000, gamma=1, kernel='poly', degree=3)
    svr.fit(X_scaled_train, y_train)

    # Predict fit for trained values
    pred_train = svr.predict(X_scaled_train)
    pred_test = svr.predict(X_scaled_test)

    # Plot.
    fig, ax = plt.subplots()
    ax.set_xlabel('Stress 1 (MPa)')
    ax.set_ylabel('Stress 2 (MPa)')
    sc = ax.scatter(X, y)
    ax.scatter(X_train, pred_train, c='r')
    ax.scatter(X_test, pred_test, c='y')
    plt.show()
