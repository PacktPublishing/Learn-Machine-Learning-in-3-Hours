import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# New libraries
from sklearn.cluster import KMeans

FILENAME = 'sample.npy'

# Load problem data.
X = np.load(FILENAME)

# Plot raw data.
fig, ax = plt.subplots()
ax.set_xlabel('Default Probability')
ax.set_ylabel('Credit Score')
ax.scatter(X[:, 0], X[:, 1])

# Check for nans and remove.
X = X[~np.isnan(X).any(axis=1)]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Above is code from previous video.
k_means = KMeans(n_clusters=4, n_jobs=-1)
k_means.fit(X_scaled)
prediction = k_means.predict(X_scaled)

h = 0.02

# Create grid to visualize clustering results.

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels at every grid point.
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot data and grid.
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_scaled[:, 0], X_scaled[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the credit score - default probability data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Default Probability')
plt.ylabel('Credit Score')
plt.xticks(())
plt.yticks(())
plt.show()



