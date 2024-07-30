# -*- coding: utf-8 -*-
"""
vis trajectory with matplotlib, then extract part of data for later use

given data with [frameid, x, y, userid, class, size]

'class' : pedestrian, car, bicycle

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



if __name__ == '__main__':
    
    
    # read csv as dataframe with Humburg dataset
    data = pd.read_csv('Hamburg_trajs.csv', sep=';', decimal=',').sort_values('fid')
    data = data[['fid', 'x', 'y', 'uid', 'class']].reset_index(drop=True)
    data['size'] = 1
    
    uids = sorted(data['uid'].unique())
    
    # plot all trajs (x<53 or x>62)
    fig, axes = plt.subplots(3,1)
    
    for uid in uids:
        user = data[(data['uid']==uid) & (data['class']=='car')]
        axes[0].plot(user['x'],user['y'],'-')
    axes[0].set_title('car')
    
    for uid in uids:
        user = data[(data['uid']==uid) & (data['class']=='person')]
        axes[1].plot(user['x'],user['y'],'-')
    axes[1].set_title('person')
    
    for uid in uids:
        user = data[(data['uid']==uid) & (data['class']=='bicycle')]
        axes[2].plot(user['x'],user['y'],'-')
    axes[2].set_title('cyclist')
    
    
    # filter take all x that is larger than 53 -> uid, filter to get branch 1
    # filter take all x that is smaller than 62 -> uid, filter to get branch 2