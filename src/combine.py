# -*- coding: utf-8 -*-
"""
vis trajectory with matplotlib 

given df with [frameid, x, y, class, userid]
solution 1: visulized with slider (s_var = frameid)
solution 2: using animation

@author: li
"""
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np

# load data
data = pd.read_pickle('../data/mapython/sportscheck_fxycu.pkl')
# load background
background_img = plt.imread('../fig/resized_sportscheck.png')
# Get unique uids and assign colors
uids = data['uid'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(uids)))
# Create figure and axes
fig, ax = plt.subplots()
ax.imshow(background_img, alpha=0.7)


''' solution 1 - slider'''

def update(fid):
    global x, y, labels, uids, alpha, colors
    # Generate new points
    new_data = data[data['fid']==fid]
    num_newpoints = len(new_data)
    new_x = new_data['x'].to_numpy()
    new_y = new_data['y'].to_numpy()
    new_labels = new_data['class']
    new_uids = new_data['uid'].to_numpy()
    
    # Append new points and labels to the existing data
    x = np.concatenate([x, new_x])
    y = np.concatenate([y, new_y])
    labels = np.concatenate([labels, new_labels])
    uids = np.concatenate([uids, new_uids])
    colors = [label_to_color[label] for label in labels]    
    # Append new alpha values and decrease the alpha of existing points
    alpha = np.concatenate([alpha * 0.5, np.ones(num_newpoints)])
    
    ax.scatter(x, y, c=colors, alpha=alpha)
    
    # Update annotations
    for i, annotation in enumerate(annotations):
        annotation.set_position((x[i], y[i]))
        annotation.set_alpha(alpha[i])
    
    # Add new annotations
    new_annotations = [ax.text(new_x[i], new_y[i], str(int(new_uids[i])), fontsize=9, ha='right', va='top', alpha=1.0) for i in range(num_newpoints)]
    annotations.extend(new_annotations)
    
    # scatter.remove()
    
# Function to update plot when slider value changes
def update_slider(val):
    frame = s_fid.val
    update(frame)
    fig.canvas.draw_idle()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')

# Plot initial frame

# extract initial data (fid == 3)
init_fid = 3
init_data = data[data['fid']==init_fid]
num_points = len(init_data)
left_fid = data['fid'].to_numpy()[1:]

x = init_data['x'].to_numpy()
y = init_data['y'].to_numpy()
uids = init_data['uid'].to_numpy()
labels = init_data['class']
alpha = np.ones(num_points)

# Map labels to colors
label_to_color = {"car": "red", "person": "green", "bicycle": "yellow"}
colors = [label_to_color[label] for label in labels]

# plot init figure
scatter = ax.scatter(x, y, c=colors, alpha=alpha)
annotations = [ax.text(x[i], y[i], str(int(uids[i])), fontsize=9, ha='right', va='top', alpha=alpha[i]) for i in range(num_points)]


# Create a Slider
ax_fid = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s_fid = Slider(ax_fid, 'Frame id', data['fid'].min(), data['fid'].max(), valinit=data['fid'].min(), valstep=1)

# Connect slider to update function
s_fid.on_changed(update_slider)

# Show plot
plt.show()

