# -*- coding: utf-8 -*-
"""
input: df trajs [uid,x,y,fid,class] and user_summary df [uid,ox,oy,ta,dx,dy,spd,class,gid(nan),wtime(0)]
output: PET
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# Example dataset: [uid, t, x, y, speed]
data = pd.read_csv('Hamburg_trajs_subset.csv')
data['speed'] = 1

# Parameters
d_threshold = 1.0  # Distance threshold for interaction
t_threshold = 5    # Maximum allowable PET

# Function to calculate PET using KD-Tree
def calculate_pet(data, d_threshold, t_threshold):
    pet_results = []
    
    # Extract relevant columns for KD-Tree
    positions = data[['x', 'y']].values
    timestamps = data['fid'].values
    uids = data['uid'].values
    
    # Build KD-Tree
    tree = KDTree(positions)
    
    # Query for neighbors within the distance threshold
    for i, (uid, pos, t) in enumerate(zip(uids, positions, timestamps)):
        indices = tree.query_radius([pos], r=d_threshold)[0]
        for j in indices:
            if i != j:  # Avoid self-comparison
                # Check temporal proximity
                if abs(t - timestamps[j]) <= t_threshold:
                    pet_results.append({
                        'uid1': uid,
                        'uid2': uids[j],
                        't1': t,
                        't2': timestamps[j],
                        'distance': np.linalg.norm(pos - positions[j]),
                        'PET': abs(t - timestamps[j])
                    })
    
    return pd.DataFrame(pet_results)

# Calculate PET
pet_df = calculate_pet(data, d_threshold, t_threshold)

# Print PET results
print("PET Results:")
print(pet_df)

# Visualization using Matplotlib
def visualize_pet(data, pet_df):
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    for uid, group in data.groupby('uid'):
        plt.plot(group['x'], group['y'], marker='o', label=f"UID {uid}")
    
    # Highlight PET events
    for _, row in pet_df.iterrows():
        x1 = data.loc[(data['uid'] == row['uid1']) & (data['fid'] == row['t1']), 'x'].values[0]
        y1 = data.loc[(data['uid'] == row['uid1']) & (data['fid'] == row['t1']), 'y'].values[0]
        x2 = data.loc[(data['uid'] == row['uid2']) & (data['fid'] == row['t2']), 'x'].values[0]
        y2 = data.loc[(data['uid'] == row['uid2']) & (data['fid'] == row['t2']), 'y'].values[0]
        
        # Draw a line between the two entities at the interaction point
        plt.plot([x1, x2], [y1, y2], color='red', linestyle='--', alpha=0.7)
        
        # Annotate PET value
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(mid_x, mid_y, f"PET: {row['PET']:.2f}", color='red', fontsize=9)
    
    # Add labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectories and Post Encroachment Time (PET)")
    plt.legend()
    plt.grid()
    plt.show()

# Generate visualization
visualize_pet(data, pet_df)

