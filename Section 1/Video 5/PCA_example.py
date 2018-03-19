import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

FILENAME = 'sample.npy'

# Load problem data.
X = np.load(FILENAME)

# Plot data set in 3D.
fig1 = plt.figure(1, figsize=plt.figaspect(0.5)*1.5)
ax1 = Axes3D(fig1)
ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
ax1.set_xlabel('West-East Position (km)')
ax1.set_ylabel('North-South Position (km)')
ax1.set_zlabel('House Prices (1000s £)')

# Fit PCA and transform.
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

# Plot transformed data.
fig2 = plt.figure(2, figsize=plt.figaspect(0.5)*1.5)
ax2 = Axes3D(fig2)
ax2.scatter(X[:, 0], X[:, 1])
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.show()

