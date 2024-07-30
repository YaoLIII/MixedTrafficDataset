# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:54:55 2024

@author: li
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: [uid, t, x, y]
# Replace this with your actual trajectory data
data = pd.DataFrame({
    'uid': [1, 1, 1, 2, 2, 2],
    't': [0, 1, 2, 0, 1, 2],
    'x': [1.0, 1.5, 1.8, 10.0, 10.5, 10.8],
    'y': [2.0, 2.5, 2.8, 12.0, 12.5, 12.8]
})

# Extract the (x, y) coordinates
coordinates = data[['x', 'y']].values

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(coordinates)

# Plotting the results
plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
plt.title('Trajectories Clustered by Location')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Print the cluster labels for each trajectory point
print(data)
