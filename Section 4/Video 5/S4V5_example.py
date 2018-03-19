import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
#New library
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


def plot_legend(ax_in):

    """Add legend to scatter plot"""

    red_patch = mpatches.Patch(color='red', label='Labour')
    blue_patch = mpatches.Patch(color='blue', label='Conservative')
    green_patch = mpatches.Patch(color='green', label='Liberal Democrat')
    ax_in.legend(handles=[red_patch, blue_patch, green_patch])


def create_grid(X, npts):

    """Create 2D of pair of input variables."""

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return np.meshgrid(np.linspace(x_min, x_max, num=npts),
                         np.linspace(y_min, y_max, num=npts))


def colormap_scatter_plot(raw_input, scaled_input, targets, fit, axis_lx, axis_ly):

    """Plot color map of fit and scatter of data"""

    # Create color maps
    cmap_l = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_b = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

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
    plot_legend(ax)
    plt.show()


if __name__ == "__main__":

    FILENAME = 'sample.npy'

    # Load problem data.
    data = np.load(FILENAME)

    # Check for error values and remove.
    data = data[np.all(data > -1, axis=1)]

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

param_grid = {
    'C': np.logspace(-3, 3),
    'gamma': np.logspace(-4, -1),
    'kernel': ['linear', 'rbf']}

svc = SVC()

clf = RandomizedSearchCV(
    svc,
    param_grid,
    cv=5,
    n_jobs=-1,
    n_iter=25)

# Fit random search model.
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

print("Best params: ", clf.best_params_)
print("Train score: ", accuracy_score(y_train, pred_train))
print("Test score: ", accuracy_score(y_test, pred_test))

# Plot
colormap_scatter_plot(X, X_train, y, clf, 'Left-Right leaning', 'Increased military spending favorability')

