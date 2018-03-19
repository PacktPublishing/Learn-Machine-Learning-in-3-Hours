import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler
from matplotlib import pyplot as plt

data = np.load('sample.npy')

# Plot raw data.
plt.figure(1)
plt.plot(data)

# Impute missing values.
imputer = Imputer()
data = imputer.fit_transform(data)

plt.figure(2)
plt.plot(data)

# Scale data.
scaler = StandardScaler()
data = scaler.fit_transform(data)

plt.figure(3)
plt.plot(data)

plt.show()

