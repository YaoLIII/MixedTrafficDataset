# -*- coding: utf-8 -*-
"""
vis trajectory with matplotlib 

given df with [frameid, x, y, class, userid]
visulized with slider (s_var = frameid)

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
    for i, user in enumerate(users):
        user_data = data[(data['uid'] == user) & (data['fid'] == frame)]
        sc[i].set_offsets(user_data[['x', 'y']])
        sc[i].set_label(user)
    
# Function to update plot when slider value changes
def update_slider(val):
    frame = s_fid.val
    update(frame)
    fig.canvas.draw_idle()
    
# load data
data = pd.read_pickle('../data/mapython/sportscheck_fxycu.pkl').head(10000)

# load background
background_img = plt.imread('../fig/resized_sportscheck.png')

# Get unique ids and assign colors
users = data['uid'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(users)))

# Create figure and axes
fig, ax = plt.subplots()
ax.imshow(background_img, alpha=0.7)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')

# Plot initial frame
sc = []
for i, user in enumerate(users):
    user_data = data[data['uid'] == user]
    s = ax.scatter(user_data['x'], user_data['y'], color=colors[i], label=user)
    # s = ax.scatter(user_data['x'], user_data['y'], color=colors[i])
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