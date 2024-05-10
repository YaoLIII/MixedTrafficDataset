# -*- coding: utf-8 -*-
"""
test vis trajectory with matplotlib 

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

# def update(f):
#     frame_data = data[data['fid'] == f]
#     sc.set_offsets(frame_data[['x', 'y']])

def update(frame):
    for i, cls in enumerate(classes):
        class_data = data[(data['class'] == cls) & (data['fid'] == frame)]
        sc[i].set_offsets(class_data[['x', 'y']])
        sc[i].set_label(cls)
    
# Function to update plot when slider value changes
def update_slider(val):
    frame = s_fid.val
    update(frame)
    fig.canvas.draw_idle()
    
# load data
data = pd.read_pickle('../data/mapython/sportscheck_fxycu.pkl')

# Get unique classes and assign colors
classes = data['class'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

# Create figure and axes
fig, ax = plt.subplots()
# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')

# # Plot initial frames
# sc = ax.scatter(data['x'], data['y'])

# Plot initial frame
sc = []
for i, cls in enumerate(classes):
    class_data = data[data['class'] == cls]
    s = ax.scatter(class_data['x'], class_data['y'], color=colors[i], label=cls)
    sc.append(s)
    
# Add legend
ax.legend()

# Create a Slider
ax_fid = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s_fid = Slider(ax_fid, 'Frame id', data['fid'].min(), data['fid'].max(), valinit=data['fid'].min(), valstep=1)

# Connect slider to update function
s_fid.on_changed(update_slider)

# Show plot
plt.show()