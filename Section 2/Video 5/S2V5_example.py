import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# New libraries
from sklearn.cluster import KMeans

FILENAME = 'sample.npy'

# Load problem data.
X = np.load(FILENAME)

# Check for nans and remove.
X = X[~np.isnan(X).any(axis=1)]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Above is code from S2V3 with plot removed.

sil_coeff = []

k_range = list(np.linspace(2, 10, 9, dtype=int))

for k in k_range:
    k_means = KMeans(n_clusters=k, n_jobs=-1)
    k_means.fit(X_scaled)
    label = k_means.labels_
    sil_coeff.append(silhouette_score(X_scaled, label, metric='euclidean'))

plt.figure(1)
plt.plot(k_range, sil_coeff)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()








