import numpy as np
from matplotlib import pyplot as plt


def plot_fit(x, y, fig_num, p_1, p_2):

    """ Plot polynomial fit """

    f = plt.figure(fig_num)
    plt.scatter(x, y)
    plt.plot(x, p_1(x))
    plt.plot(x, p_2(x))
    plt.show()


data = np.load('sample.npy')

# Split data into training and testing sets.
train_test_ratio = 0.5

len_data = len(data[0, :])

train_end = int(train_test_ratio * len_data)

train_data = data[:, :train_end]
test_data = data[:, train_end:]

# Fit polynomials
poly_1 = np.polyfit(train_data[0, :], train_data[1, :], 1)
poly_2 = np.polyfit(train_data[0, :], train_data[1, :], 2)

poly_1 = np.poly1d(poly_1)
poly_2 = np.poly1d(poly_2)

plot_fit(train_data[0, :], train_data[1, :], 1, poly_1, poly_2)

plot_fit(test_data[0, :], test_data[1, :], 1, poly_1, poly_2)

future_data = np.load('future_data.npy')

plot_fit(future_data[0, :], future_data[1, :], 1, poly_1, poly_2)
