# -*- coding: utf-8 -*-
"""
extract part of data for later use, by given fid range [,] to get the users exists fully inside the range

input:
    given data with [frameid, x, y, userid, class, size]
    'class' : pedestrian, car, bicycle

params: frange[f_start, f_duration]
output: sub-data table with [uid,x,y,fid,class] and user_summary table [uid,ox,oy,ta,dx,dy,spd,class,gid(nan),wtime(0)]

@author: li
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def selected_data(df, f_start = 100, f_duration = 200):
    print("start extracting data ...")
    
    f_end = f_start + f_duration
    
    # existanse table [uid, fmin, fax]
    user_exist = df.groupby('uid')['fid'].agg(['min','max']).reset_index()
    user_exist.columns = ['uid','min_fid','max_fid']
    
    # Filter users who are fully within the specified frame range
    extracted_uid = user_exist[(user_exist['min_fid'] >= f_start) & (user_exist['max_fid'] <= f_end)]['uid']
    
    # output extracted trajectories and user_summary
    selected_df = df[df['uid'].isin(extracted_uid)].sort_values(by=['uid', 'fid']).reset_index(drop=True)
    selected_user_summary = selected_df.groupby('uid').apply(lambda group: pd.Series({
        'ox': group['x'].iloc[0],
        'oy': group['y'].iloc[0],
        'ta': group['fid'].iloc[0],
        'dx': group['x'].iloc[-1],
        'dy': group['y'].iloc[-1],
        'spd': group['spd'].mean(),
        'class': group['class'].iloc[0]
    })).reset_index()
    
    print('selection done!')
    
    # add gid and wtime into user_summery
    selected_user_summary['gid'] = np.nan
    selected_user_summary['wtime'] = 0
    
    return selected_df, selected_user_summary

# def calculate_average_speed(group):
#     # Calculate total distance
#     distances = np.sqrt(np.diff(group['x'])**2 + np.diff(group['y'])**2)
#     total_distance = distances.sum()
    
#     # Calculate time duration
#     time_duration = group['fid'].iloc[-1] - group['fid'].iloc[0]
    
#     # Handle division by zero if time_duration is zero
#     average_speed = total_distance / time_duration if time_duration > 0 else 0
#     return average_speed


if __name__ == '__main__':    
    
    # read csv as dataframe with Humburg dataset
    df = pd.read_csv('Hamburg_trajs.csv', sep=';', decimal=',').sort_values(by=['uid', 'fid'])
    df = df[['uid', 'x', 'y', 'fid', 'spd', 'class']].reset_index(drop=True).dropna(subset=['spd'])
    
    # the maxmium value of current frame id, to restrict 
    max_fid_start = df['fid'].max() 
    # take variables from input
    f_start = int(input("Enter desired start frame (1-" + str(max_fid_start) + "):"))
    f_duration = int(input("Enter desired traffic duration (0-" + str(max_fid_start - f_start) + "):"))
    
    trajs, user_summary = selected_data(df, f_start, f_duration)
    
    # save the extracted data
    trajs.to_csv('Hamburg_trajs_subset.csv', index=False)
    user_summary.to_csv('Hamburg_user_summary_subset.csv', index=False)
    
    