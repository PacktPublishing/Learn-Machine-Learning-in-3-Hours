import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def create_grid(X, npts):

    """Create 2D of pair of input variables."""

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return np.meshgrid(np.linspace(x_min, x_max, num=npts),
                         np.linspace(y_min, y_max, num=npts))


def colormap_scatter_plot(raw_input, scaled_input, targets, fit, axis_lx, axis_ly):

    """Plot color map of fit and scatter of data"""

    # Create color maps
    cmap_l = 'Pastel2'
    cmap_b = 'Dark2'

    # Create grid of scaled variables to predict labels.

    xx_scaled, yy_scaled = create_grid(scaled_input, 50)

    Z = fit.predict(np.c_[xx_scaled.ravel(), yy_scaled.ravel()])

    Z = Z.reshape(xx_scaled.shape)

    # Create grid for plotting.

    xx, yy = create_grid(raw_input, 50)

    fig, ax = plt.subplots()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_l)

    # Plot also the training points
    plt.scatter(raw_input[:, 0], raw_input[:, 1], c=targets, cmap=cmap_b, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax.set_xlabel(axis_lx)
    ax.set_ylabel(axis_ly)
    plt.show()


if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Check for error values and remove.

    # Split into input data and labels.
    y = data[:, -1]
    X = data[:, :-1]

    # Scale data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Generate training and testing set.
    mid_point = int(len(X_scaled) / 2)

    X_train = X_scaled[:mid_point, :]
    X_test = X_scaled[mid_point:, :]
    y_train = y[:mid_point]
    y_test = y[mid_point:]

    # Above is code from last video.

    # Fit gradient boosting model.
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)

    pred_train = gbc.predict(X_train)
    pred_test = gbc.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred_train, y_train))
    print(accuracy_score(pred_test, y_test))

    # Plot
    colormap_scatter_plot(X, X_train, y, gbc, 'Asset 1 Price Change', 'Asset 2 Price Change')
