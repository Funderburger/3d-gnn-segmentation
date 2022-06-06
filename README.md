# 3d-gnn-segmentation


## (Conda) Environment setup:
- Python 3.8
- Pytorch 1.11.0+cu113 (via pip install)
- PyG 1.11.0+cu113 (2.0.4) (via pip install)
- Open3D 0.15.2 (via pip install)
- Trimesh

## Catkin Environment Setup:
- `catkin_don_ws` is your `catkin_ws`
- just hit a `catkin_make` in there and you **should** be good to go
- if you have any erros, well RIP! (though, you can open an issue, as I **_might_** be able to help you)

## How I created the data, you ask? (Ez!)
- first of all, I took the SUNRGBD data from here (https://rgbd.cs.princeton.edu) using `wget`:
  - `wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip`
- and their toolbox:
  - `wget https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip`
- and... I made a _few_ changes as it can be seen in the SUNRGBDtoolbox_MOD, in order to create two sets of point clouds in two different folders:
  - `pcd_rgb_data`
  - `pcd_label_data`
- by the way, I used the following command to run the matlab script from terminal:
  -  `matlab -nodisplay -nosplash -nodesktop -r "run('demo_full_pcd.m');exit;"`
- then, I used the `down_norm_fpfh` in order to create downsampled point clouds (!!beware thogh, because the number specified as an argument for downsampling is actually with 10k points greater than the actual number, as youn can see in the code, because normal estimation will always result in a few NaN points, that is why it is better to set a higher number and the last downsample will take care of it in order to acquire a fixed number of points, even though you should never do what I did here as this is very misleading) (also, you should have `pcd_rgb_data` and `pcd_label_data` folders in the same folder):
  - `catkin_don_ws/devel/lib/don_segmento/down_norm_fpfh -input_dir /home/marian/workspace/gnns_ws/don_ws/SUNRGBDtoolbox_MOD/pcd_rgb_data -output_dir /home/marian/workspace/gnns_ws/don_ws/SUNRGBDtoolbox_MOD/pcd_data_combo_fpfh -radius 0.1 -k 0 -samples 50000 -fpfh_param 0.2`
- after that you can load your data, as I did in the `process` method from `MyOwnDataset` (gnn_stuff/train_gnn.py), but in order to work you should have in your `root` path variable two folders named: `processed` and `raw` and in your raw directory (or symlink ;) ) you need to have the previous obtained point clouds which you can process after your own will. 
