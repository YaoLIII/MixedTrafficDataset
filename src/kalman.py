# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:15:47 2024

@author: li
"""

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# Simulated 2D trajectory with noise
np.random.seed(123)
n_samples = 100
t = np.linspace(0, 10, n_samples)
x_true = 2 * np.sin(t)
y_true = 2 * np.cos(t)
x_noisy = x_true + np.random.normal(0, 0.5, n_samples)
y_noisy = y_true + np.random.normal(0, 0.5, n_samples)

# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)

# Define state transition matrix
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Define measurement function
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

# Define process noise covariance
kf.Q *= 0.01

# Define measurement noise covariance
kf.R *= 0.5

# Initialize state estimate
kf.x = np.array([x_noisy[0], y_noisy[0], 0, 0])

# Initialize state covariance
kf.P *= 0.1

# Smooth the trajectory using Kalman filter
smoothed_trajectory = []
for z in zip(x_noisy, y_noisy):
    kf.predict()
    kf.update(z)
    smoothed_trajectory.append((kf.x[0], kf.x[1]))

smoothed_x, smoothed_y = zip(*smoothed_trajectory)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_noisy, y_noisy, 'bo', label='Noisy Trajectory')
plt.plot(smoothed_x, smoothed_y, 'r-', label='Smoothed Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter Smoothing of 2D Trajectory')
plt.legend()
plt.grid(True)
plt.show()

