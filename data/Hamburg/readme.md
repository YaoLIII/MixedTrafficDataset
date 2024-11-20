# Trajectories from Sahre space in Hamburg
## Contents
Folder contents
```bash
├── Hamburg_dataset.csv
├── Hamburg_trajs.csv
├── Hamburg_trajs_en.csv
├── background_Bergedorf.jpg
├── background_Bergedorf_resize2.jpg
├── background_Bergedorf_resized.jpg
├── devide_trajectories.py
├── markdown_structure.py
├── readme.md
├── test_anim.mp4
├── vis_data_extract.py
└── visualization1.py
```
based on raw data, it:

1. changed colnames
2. change type/class names
3. minus 10000 to uids
4. change "bicycle   " to "bicycle", e.g.

## Functions 
### data_extract.py
take part of the trajectory data as samples for experiement.
[input] df = [fid, x, y, uid, class, size], class includes "pedestrian, car, bicycle"
[param] desired frame range = [f_start, f_duration]
[output] sub-trajectory df [uid,x,y,fid,class] and user_summary df [uid,ox,oy,ta,dx,dy,spd,class,gid(np.nan),wtime(0)]

### visualization.py
take any df with [fid, x, y, uid, class] and visualize it as animation - as the vis of original movement, used to compare with new data

### markdown_structure.py
simply generate folder structure for in markdown format.

### 