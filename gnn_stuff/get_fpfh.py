import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 10
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def main():
    pcd_name = 'SUNRGBDtoolbox/pcd_data_testing/sunrgbd_xyzrgbi_1.txt'
    # label_pcd_name = 'SUNRGBDtoolbox/pcd_label_data/sunrgbd_pcd_1.pcd'
    # rgb_pcd = o3d.io.read_point_cloud(rgb_pcd_name, format='pcd')
    # label_pcd = o3d.io.read_point_cloud(label_pcd_name, format='pcd')
    # o3d.visualization.draw_geometries([rgb_pcd])
    pcd_np = np.genfromtxt(pcd_name, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(pcd_np[:,3:6]/255)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    vox_normal_pcd, fpfh_pcd = preprocess_point_cloud(pcd,0.001)

    # o3d.visualization.draw_geometries([label_pcd])
    # o3d.io.write_point_cloud(pcd_name[:-4] +"_pcl.pcd" , pcd)

if __name__=='__main__':
    main()