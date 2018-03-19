import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# New libraries
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, 0]

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
    X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.5, random_state=42)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.5, random_state=42)

    # Above is code from previous video
    param_grid = {
            'C': np.logspace(0, 5),
            'gamma': np.logspace(-2, 1, 3),
            'degree': np.linspace(2, 7, 6)}

    svr = SVR(kernel='poly', max_iter=10000)

    clf = RandomizedSearchCV(
        svr,
        param_grid,
        cv=5,
        n_jobs=-1,
        n_iter=200)

    # Fit random search model.
    clf.fit(X_scaled_train, y_train)
    pred_train = clf.predict(X_scaled_train)
    pred_test = clf.predict(X_scaled_test)

    print("Best params: ", clf.best_params_)
    print("Train score: ", mean_squared_error(y_train, pred_train))
    print("Test score: ", mean_squared_error(y_test, pred_test))

    # Plot.
    fig, ax = plt.subplots()
    ax.set_xlabel('Stress 1 (MPa)')
    ax.set_ylabel('Stress 2 (MPa)')
    sc = ax.scatter(X, y)
    ax.scatter(X_train, pred_train, c='r')
    ax.scatter(X_test, pred_test, c='y')
    plt.show()





