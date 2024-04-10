# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:15:47 2024

@author: li
"""

# pip install pykalman

import numpy as np
from pykalman import KalmanFilter

# Generate some sample noisy 2D trajectory data
np.random.seed(0)
n_samples = 100
true_positions = np.linspace(0, 10, n_samples)
noisy_positions = true_positions + np.random.normal(0, 1, (n_samples, 2))

# Create a Kalman filter object
kf = KalmanFilter()

# Smooth the trajectory data using the Kalman filter
smoothed_positions, _ = kf.smooth(noisy_positions)

# Plot the noisy and smoothed trajectories
import matplotlib.pyplot as plt
plt.plot(true_positions, label='True trajectory')
plt.plot(noisy_positions[:, 0], noisy_positions[:, 1], label='Noisy trajectory')
plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], label='Smoothed trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Trajectory Smoothing with Kalman Filter')
plt.legend()
plt.show()
