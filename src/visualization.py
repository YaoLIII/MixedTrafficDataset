# -*- coding: utf-8 -*-
"""
test vis trajectory with matplotlib 

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# load data
data = pd.read_pickle('../data/mapython/sportscheck_fxycu.pkl')

# TODO: The parametrized function to be plotted
def extract_user(f, data):
    fdata = data[data['fid']==f]
    return fdata

frame = data['fid']

# TODO: Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
fdata = extract_user(frame, data)
line, = ax.plot(fdata['x'],fdata['y'])
ax.set_xlabel('Frame id')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(
    ax=axframe,
    label='Frame',
    valmin=3,
    valmax=max(frame),
    valinit=3,
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(extract_user(frame, data))
    fig.canvas.draw_idle()


# register the update function with each slider
frame_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    frame_slider.reset()
button.on_clicked(reset)

plt.show()