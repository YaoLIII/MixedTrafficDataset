# -*- coding: utf-8 -*-
"""
input: df trajs [uid,x,y,fid,class] 
       user_summary df [uid,ox,oy,ta,dx,dy,spd,class,gid(nan),wtime(0)]
output: PET 

"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# dataset: [uid, x, y, fid, spd, class]
data = pd.read_csv('Hamburg_trajs_subset.csv')

# PET Parameters
d_threshold = 1.0  # Distance threshold for interaction
t_threshold = 0.5    # Maximum allowable PET

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


'''method2'''
import pandas as pd
from shapely.geometry import LineString, box
from rtree import index

# dataset: [uid, x, y, fid, spd, class]
data = pd.read_csv('Hamburg_trajs_subset.csv')

# each trajectory saved as shapely.LingString
trajs_LS = data.groupby('uid').apply(
    lambda group: LineString(zip(group['x'], group['y']))
)

# insert traj bbox into the R-tree
r_tree = index.Index()
for uid, trajectory in trajs_LS.items():
    r_tree.insert(uid, trajectory.bounds)

# Find conflicts using R-tree
conflicts = []
for uid, traj in trajs_LS.items():
    # Query the R-tree for possible intersections
    possible_matches = list(r_tree.intersection(traj.bounds))
    for match_id in possible_matches:
        if uid != match_id:  # no self-comparison
            other_traj = trajs_LS[match_id]
            if traj.intersects(other_traj):
                conflicts.append((uid, match_id))

# Remove duplicate conflicts (e.g., (1, 2) and (2, 1))
unique_conflicts = set(tuple(sorted(conflict)) for conflict in conflicts)

# Output conflicts
conflict_data = pd.DataFrame(unique_conflicts, columns=['uid_1', 'uid_2'])
print(conflict_data)

'''trajectory orientations'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 轨迹点集
trajectory = np.array([
    [0, 0],
    [1, 1],
    [2, 1.5],
    [3, 2],
    [4, 3]
])

# 使用PCA计算主要方向
pca = PCA(n_components=2)
pca.fit(trajectory)

# 第一主成分方向
principal_direction = pca.components_[0]
angle = np.arctan2(principal_direction[1], principal_direction[0])

# 轨迹中心点作为箭头起点
center = trajectory.mean(axis=0)

# 可视化轨迹与主要方向
plt.figure(figsize=(8, 6))

# 绘制轨迹
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label="Trajectory", markersize=8)

# 绘制主要方向箭头
plt.quiver(
    center[0], center[1],  # 箭头起点
    principal_direction[0], principal_direction[1],  # 箭头方向
    angles='xy', scale_units='xy', scale=1, color='r', label="Principal Direction"
)

# 图形美化
plt.title("Trajectory and Principal Direction")
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.legend()
plt.axis('equal')  # 保持比例一致
plt.show()



