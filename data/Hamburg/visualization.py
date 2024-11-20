#-*- coding: utf-8 -*-
"""
vis trajectory with matplotlib 

given df with [frameid, x, y, class, userid]
solution 1: visulized with slider (s_var = frameid)
solution 2: using animation

@author: li
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

# read csv as dataframe with Humburg dataset
data = pd.read_csv('Hamburg_trajs_subset.csv', sep=',', decimal='.').sort_values('fid')
data = data[['fid', 'x', 'y', 'uid', 'class']].reset_index(drop=True)
data['size'] = 1

# load background
background_img = plt.imread('background_Bergedorf_resized.jpg')
# Get unique uids and assign colors
uids = data['uid'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(uids)))
# Create figure and axes
fig, ax = plt.subplots()
ax.imshow(background_img, alpha=0.7)

# extract initial data (fid == 3)
init_fid = min(data['fid'])
init_data = data[data['fid']==init_fid]
num_points = len(init_data)
left_fid = np.unique(data['fid'].to_numpy()[1:])

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

def update(fid):
    global x, y, labels, uids, alpha, colors, scatter
    
    print(fid) # check
    ax.set_title(u"movement at frame {}".format(str(int(fid))))
    scatter.remove()
    
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
    alpha = np.concatenate([alpha * 0.8, np.ones(num_newpoints)])
   
    # Redraw the scatter plot data
    scatter = ax.scatter(x, y, c=colors)
    scatter.set_alpha(alpha)
    
    alpha_annotations = np.zeros(len(alpha))
    alpha_annotations[-1] = 1
    
    # Update annotations
    for i, annotation in enumerate(annotations):
        annotation.set_position((x[i], y[i]))
        annotation.set_alpha(alpha_annotations[i])
    
    # Add new annotations
    new_annotations = [ax.text(new_x[i], new_y[i], str(int(new_uids[i])), fontsize=9, ha='right', va='top', alpha=1.0) for i in range(num_newpoints)]
    annotations.extend(new_annotations)
    
    return scatter, *annotations

# Create animation
ani = animation.FuncAnimation(fig, update, frames=left_fid, interval=0.1, blit=False)

start_time = time.time()
ani.save('hamburg_subset_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
end_time = time.time()

print("--- %s mins ---" % (time.time() - start_time))

# # Show the plot
# plt.show()


