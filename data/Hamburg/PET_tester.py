import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from rtree import index
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_trajectories_with_timestep(df):
    # Create a color map for time steps (fid)
    time_steps = df['fid'].unique()
    norm = mcolors.Normalize(vmin=min(time_steps), vmax=max(time_steps))
    cmap = cm.viridis

    plt.figure(figsize=(12, 8))

    for uid in df['uid'].unique():
        traj = df[df['uid'] == uid]
        
        # Plot trajectory with a color gradient based on fid
        sc = plt.scatter(traj['x'], traj['y'], c=traj['fid'], cmap=cmap, norm=norm, s=10, label=f"UID {uid}")

    # Add a color bar for time steps
    cbar = plt.colorbar(sc)
    cbar.set_label('Time Step (fid)', rotation=270, labelpad=15)

    plt.title("Trajectory Movements with Time Step Bar")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
def plot_PET_hist(df, dataset_name='dataset'):
    # df [uid1,uid2,fid, dist, PET_min]
    plt.figure()
    bins = int(len(df)/2)
    plt.hist(df['PET_min'], bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of PET_min in {dataset_name}' )
    plt.xlabel('PET_min')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
# Load random n users' trajs
data = pd.read_csv('Hamburg_trajs_subset.csv')
unique_uids = data['uid'].unique()
random_uids = pd.Series(unique_uids).sample(n=30, random_state=42).tolist()
df = data[data['uid'].isin(random_uids)]

plot_trajectories_with_timestep(df)

def check_temporal_coexistence(df):
    temporal_conflicts = []
    unique_uids = df['uid'].unique()
    
    for i, uid1 in enumerate(unique_uids):
        for uid2 in unique_uids[i + 1:]:
            traj1 = df[df['uid'] == uid1]
            traj2 = df[df['uid'] == uid2]
            
            # Check temporal overlap
            t1_min, t1_max = traj1['fid'].min(), traj1['fid'].max()
            t2_min, t2_max = traj2['fid'].min(), traj2['fid'].max()
            
            if t1_max >= t2_min and t2_max >= t1_min:  # Overlapping time ranges
                temporal_conflicts.append((uid1, uid2))
    
    return temporal_conflicts

def calculate_angle_difference(traj1, traj2):
    # PCA to calculate main orientation
    pca1 = PCA(n_components=1).fit(traj1)
    pca2 = PCA(n_components=1).fit(traj2)
    angle1 = np.arctan2(pca1.components_[0][1], pca1.components_[0][0])
    angle2 = np.arctan2(pca2.components_[0][1], pca2.components_[0][0])
    return np.abs(np.degrees(angle1 - angle2))

def check_spatial_conflict(df, temporal_conflicts, spatial_threshold=0.5):
    spatial_conflicts = []
    # check the orientation difference by intersection and PCA
    for uid1, uid2 in temporal_conflicts:
        traj1 = df[df['uid'] == uid1]
        traj2 = df[df['uid'] == uid2]
        
        # Find overlapping time frames
        overlapping_fids = set(traj1['fid']).intersection(set(traj2['fid']))
        
        for fid in overlapping_fids:
            point1 = traj1[traj1['fid'] == fid][['x', 'y']].values[0]
            point2 = traj2[traj2['fid'] == fid][['x', 'y']].values[0]
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(point1 - point2)
            if distance <= spatial_threshold:
                spatial_conflicts.append((uid1, uid2, fid, distance))
    
    return spatial_conflicts

def PET(df, spatial_conflicts, th_PET=0.5):
    columns = ['uid1', 'uid2', 'fid', 'distance']
    spatial_conflicts_df = pd.DataFrame(spatial_conflicts, columns=columns)
    
    min_distances = (
        spatial_conflicts_df
        .groupby(['uid1', 'uid2'], as_index=False)
        .apply(lambda group: group.loc[group['distance'].idxmin()])
        .reset_index(drop=True)
    )
    
    # Merge to get spd for uid1
    merged1 = pd.merge(min_distances, df[['uid', 'fid', 'spd']], left_on=['uid1', 'fid'], right_on=['uid', 'fid'])
    merged1.rename(columns={'spd': 'spd_user1'}, inplace=True)
    merged1.drop(columns=['uid'], inplace=True)
    
    # Merge to get spd for uid2
    merged2 = pd.merge(merged1, df[['uid', 'fid', 'spd']], left_on=['uid2', 'fid'], right_on=['uid', 'fid'])
    merged2.rename(columns={'spd': 'spd_user2'}, inplace=True)
    merged2.drop(columns=['uid'], inplace=True)
    
    # Calculate PET1, PET2, and PET_min
    merged2['PET1'] = merged2['distance'] / merged2['spd_user1']
    merged2['PET2'] = merged2['distance'] / merged2['spd_user2']
    merged2['PET_min'] = merged2[['PET1', 'PET2']].min(axis=1)
    
    # Create the final DataFrame and filter it with th_PET<0.5 (dangerous)
    result = merged2[['uid1', 'uid2', 'fid', 'distance', 'PET_min']]
    result_dangerous = result[result['PET_min'] < th_PET]
    
    # plot dangerous histogram
    plot_PET_hist(result, dataset_name='result')
    plot_PET_hist(result_dangerous, dataset_name='result_dangerous')
    
    # plot dangerous_result
    
    
    return result, result_dangerous

# Step 1: Check Temporal Co-Existence
temporal_conflicts = check_temporal_coexistence(df)

# Step 2: Check Spatial Conflict
spatial_conflicts = check_spatial_conflict(df, temporal_conflicts, spatial_threshold=5)

# Step 3: Take the most dangerous moment and calculate PET
result, result_dangerous = PET(df, spatial_conflicts, th_PET=0.5)

# Step 4: Display Results
def plot_trajectories(result_dangerous, df):
    """
    Plots the trajectories of each pair of users (uid1, uid2) from result_dangerous.
    
    Parameters:
        result_dangerous (DataFrame): DataFrame with columns [uid1, uid2, fid, distance, PET_min].
        df (DataFrame): DataFrame with columns [uid, x, y, fid, spd, class].
    """
    # Generate unique colors for each pair
    unique_pairs = result_dangerous[['uid1', 'uid2']].drop_duplicates()
    colors = plt.cm.tab20.colors  # Use a colormap for colors
    color_map = {tuple(pair): colors[i % len(colors)] for i, pair in enumerate(unique_pairs.values)}

    plt.figure(figsize=(10, 8))

    # Plot trajectories for each pair
    for _, row in result_dangerous.iterrows():
        uid1, uid2 = row['uid1'], row['uid2']

        # Extract trajectories for uid1 and uid2
        traj1 = df[df['uid'] == uid1]
        traj2 = df[df['uid'] == uid2]

        # Get the assigned color for the pair
        pair_color = color_map[(uid1, uid2)]

        # Plot trajectories
        plt.plot(traj1['x'], traj1['y'], label=f'User {uid1}', color=pair_color, alpha=0.7)
        plt.plot(traj2['x'], traj2['y'], label=f'User {uid2}', color=pair_color, alpha=0.7, linestyle='dashed')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of Conflict Users')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.5)
    plt.show()

plot_trajectories(result_dangerous, df)




