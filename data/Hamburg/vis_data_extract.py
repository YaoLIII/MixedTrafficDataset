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
    
    # plot all trajs
    for uid in uids:
        user = data[data['uid']==uid]
        plt.plot(user['x'],user['y'],'o-')

    
    