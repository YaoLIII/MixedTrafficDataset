# -*- coding: utf-8 -*-
"""
extract part of data for later use, by given fid range [,] and the users exists fully inside the range

input:
    given data with [frameid, x, y, userid, class, size]
    'class' : pedestrian, car, bicycle

params: frange[f_start, f_duration]
output: sub-data table with [uid,x,y,fid,class] and userinfo table [uid,ox,oy,ta,dx,dy,spd,class,gid,wtime]

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def HamburgSubData(df, f_start = 1, f_duration = 200):

if __name__ == '__main__':
    
    
    # read csv as dataframe with Humburg dataset
    data = pd.read_csv('Hamburg_trajs.csv', sep=';', decimal=',').sort_values('uid')
    data = data[['uid', 'x', 'y', 'fid', 'class']].reset_index(drop=True)
    
    