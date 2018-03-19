import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

fig2, ax2 = plt.subplots()
ax2.set_xlabel('Default Probability')
ax2.set_ylabel('Credit Score')
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1])

plt.show()




