# -*- coding: utf-8 -*-
"""
deal with tracking results

1. reorgnizing tracking results by assigning unique uid according to classes
2. generating userInfo and trajs for later use
3. visualizing tracking results (by matplotlib slider) - dropped
4. manully work: replace ped+bi with cyc

@author: li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_reorganizer(path='../data/mapython/transformed_data_np.npy'):
    '''
    change the tracking trajectories by:
        - make uid unique according to classes

    ----------
    i : [frame, cx, cy, class_id, uid]
    o : [frame, cx, cy, class_id, unique_uid]
    -------


    '''
    data = np.load(path)
    mapping = {'person': 1, 'bicycle': 2, 'car': 3}
    
    # replace class str by index
    df = pd.DataFrame(data, columns=['fid','cx','cy','class','uid'])
    df['class'] = df['class'].map(mapping) 
    # convert str to num
    df['fid'] = df['fid'].astype(int)
    df['uid'] = df['uid'].astype(int)
    df['cx'] = df['cx'].astype(float)
    df['cy'] = df['cy'].astype(float)
    
    # extract users by class
    df_ped = df[df['class']==1]
    df_byc = df[df['class']==2]
    df_car = df[df['class']==3]
    
    # count the number of each class
    uid_ped = pd.unique(df_ped['uid'])
    uid_byc = pd.unique(df_byc['uid'])
    uid_car = pd.unique(df_car['uid'])
    print('there are '+ str(len(uid_ped))+' ped, ' + str(len(uid_byc)) + \
          ' bike, and ' + str(len(uid_car)) + ' car in current data.')
        
    # map new uid for different class (make uid unique)
    new_uid_ped = np.arange(0, 0 + len(uid_ped))
    new_uid_byc = np.arange(new_uid_ped[-1]+1, new_uid_ped[-1]+1 + len(uid_byc))
    new_uid_car = np.arange(new_uid_byc[-1]+1, new_uid_byc[-1]+1 + len(uid_car))
    
    new_car = replaceDfValues(df_car, uid_car, new_uid_car, 'uid').to_numpy()
    new_byc = replaceDfValues(df_byc, uid_byc, new_uid_byc, 'uid').to_numpy()
    new_ped = replaceDfValues(df_ped, uid_ped, new_uid_ped, 'uid').to_numpy()
    
    new_data =  np.concatenate((new_car, new_byc, new_ped))
    
    return new_data

def replaceDfValues(df, oldc, newc, col='uid'):
    '''
    replace a certain column of df with new values
    '''
    mapping = dict(zip(oldc,newc))
    # df[col] = df[col].map(mapping)
    df.loc[:,col] = df.loc[:,col].map(mapping)
    
    return df

def dataConvert(data):
    '''
    from tracking results to userInfo and Trajs

    ----------
    i : [frame, cx, cy, class_id, uid]
    o : [id ox oy t_a dx dy speed type gid_0, wtime_0]
       - userInfo [0:uid 1:ox 2:oy 3:t_a 4:dx 5:dy 6:speed 7:class 8:gid(0) 9:w_time(0) ]
       - trajs    [0:fid 1:uid 2:x 3:y 4:type]

    ''' 
    
    # reorder to get trajs
    col_indices = [0, 4, 1, 2, 3]
    trajs = data[:, col_indices]
    # sort_indices = np.argsort(trajs[:, 0])
    # trajs = trajs[sort_indices]
    
    # extract userInfo from trajs
    userInfo = []
    for uid in np.unique(trajs[:,1]):
        user = trajs[trajs[:,1]==uid]
        vel = avgVel(user)
        # [id,ox,oy,t0,dx,dy,speed,gid,w_time,type]
        user_info = [uid,user[0,2],user[0,3],user[0,0],user[-1,2],user[-1,3], vel, user[0,4], 0, 0]
        userInfo.append(user_info)
    userInfo = np.asarray(userInfo)
    
    return trajs, userInfo

def dataConvert_df(df):
    
    return True

def avgVel(user):
    '''
    calculate avg velocity of a user given [fid uid x y type]
    '''
    diff_dist = np.diff(user[:, 2:4], axis=0)
    dist = np.sum(np.linalg.norm(diff_dist, axis=1))
    
    duration = user[-1, 0] - user[0, 0]
    
    vel = dist/duration 
    
    return vel

if __name__ == "__main__":
    
    print("reassigning unique ids to tracking data...")
    data = data_reorganizer()    
    # np -> df
    cols = ['fid','x','y','class','uid']
    df = pd.DataFrame(data, columns=cols)    
    mapping_nts = {1:'person', 2: 'bicycle', 3:'car'}
    df['class'] = df['class'].map(mapping_nts)
    
    # save as csv for manully checking - might still need to change in excel due to German typing style...
    print("Save as csv for trajectory correction...")
    df.to_csv('../data/mapython/sportscheck_fxycu.csv', sep=',', encoding='utf-8', index=False, decimal=',')
    input("Press Enter after correction in csv file...")
    
    print("Saving filtered trajs as userInfo...")
    df_filtered = pd.read_csv('../data/mapython/sportscheck_fxycu_0625.csv', sep=';', decimal=',').sort_values('fid')
    # df->np (class col string -> num)
    mapping_stn = {'person': 1, 'bicycle': 2, 'car': 3}
    df_filtered['class'] = df_filtered['class'].map(mapping_stn)
    data_filtered = df_filtered.to_numpy()    
    trajs, userInfo = dataConvert(data_filtered)
    np.save('D:/Seafile/moveagentinformation/data/mapython_userInfo.npy', userInfo)
    np.save('D:/Seafile/moveagentinformation/data/mapython_trajs.npy', trajs)
