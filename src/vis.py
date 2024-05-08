# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:11:34 2024

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'timestep': [3.0, 4.0, 5.0, 6.0, 7.0],
    'x': [1611.14300, 1611.14300, 1611.14300, 1611.14300, 1611.14300],
    'y': [765.25620, 765.25620, 765.25620, 765.25620, 765.25620],
    'class': ['car', 'car', 'car', 'car', 'car'],
    'trajectory_id': [99.0, 99.0, 99.0, 99.0, 99.0]
}
df = pd.DataFrame(data)

# Create figure and axes
fig, ax = plt.subplots()

# Define colors for different classes
class_colors = {'car': 'blue', 'bicycle': 'green', 'person': 'red'}

# Plot initial frame
sc = ax.scatter(df['x'], df['y'], c=df['class'].map(class_colors))

# Add legend for classes
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name, markersize=10, markerfacecolor=color)
                   for class_name, color in class_colors.items()]
ax.legend(handles=legend_elements, loc='upper right')

# Function to annotate trajectories
def annotate_trajectory(trajectory_id, x, y):
    ax.annotate(str(trajectory_id), (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Add trajectory annotations
for index, row in df.iterrows():
    annotate_trajectory(row['trajectory_id'], row['x'], row['y'])

# Add slider
ax_timestep = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s_timestep = Slider(ax_timestep, 'Timestep', df['timestep'].min(), df['timestep'].max(), valinit=df['timestep'].min(), valstep=1)

# Update function for animation
def update(frame):
    frame_data = df[df['timestep'] == frame]
    sc.set_offsets(frame_data[['x', 'y']])
    sc.set_color(frame_data['class'].map(class_colors))

# Function to update plot when slider value changes
def update_slider(val):
    frame = s_timestep.val
    update(frame)
    fig.canvas.draw_idle()

# Connect slider to update function
s_timestep.on_changed(update_slider)

# Show plot
plt.show()
