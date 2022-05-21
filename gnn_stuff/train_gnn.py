from logging import root
from turtle import pos
import open3d as o3d
import numpy as np

from time import sleep
from pathlib import Path
from itertools import tee
from functools import lru_cache

import trimesh
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import BaseTransform, Compose, FaceToEdge, KNNGraph, RadiusGraph
from torch_geometric.data import Data, InMemoryDataset, extract_zip, DataLoader, Dataset, download_url

import os
import os.path as osp

from pyntcloud import PyntCloud

import scipy.io
# mat = scipy.io.loadmat('SUNRGBDtoolbox/Metadata/seglistall.mat')
# mat['seglistall'][0][5]
torch.cuda.set_device(1)

class MyOwnDataset(Dataset):
    def __init__(self, root,transform=None, pre_transform=None, pre_filter=None):
        # self.test = test
        # self.
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        files = []
        for filename in sorted(os.scandir(self.raw_dir), key=lambda f: f.name):
            files.append(filename.name)
        return files

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        # files = []
        # for filename in self.raw_paths:
        #     files.append(filename)
        self.data = self.raw_paths

        # if self.test:
        #     return [f'data_test_{i}.pt' for i in self.data]
        # else:
        #     return [f'data_{i}.pt' for i in self.data]
        return [f'data_{i}.pt' for i in self.data]

    def download(self):
        pass

    def process(self):
        # idx = 0
        # files = []
        # for filename in sorted(os.scandir(self.raw_paths[0]), key=lambda f: f.name):
        #     files.append(filename)
        self.data = self.raw_paths
        index = 1
        for pcd_name in tqdm(self.data, total=self.data.__len__()):
            # pcd_np = np.genfromtxt(pcd_name, delimiter=",")
            pcd_object = PyntCloud.from_file(pcd_name)
            pcd_np = np.array([pcd_object.points.x, pcd_object.points.y, pcd_object.points.z, pcd_object.points.label]).T

             # Get node features
            # node_feats = torch.tensor(np.concatenate([pcd_np[:,:3], pcd_np[:,3:6]/255],1)).cuda()
            node_feats = torch.tensor(pcd_np[:,:3]).cuda()
            # Get edge features
            edge_feats = None
            # Get adjacency info
            torch_points = torch.tensor(pcd_np[:,:3]).cuda()
            edge_index = torch_geometric.nn.knn_graph(
                torch_points,
                3,
                None,
                loop=False,
                flow='source_to_target',
                cosine=False,
                num_workers=1,
            )
            # Get labels info
            # label = self._get_labels(pcd_np[:,6])
            label = self._get_labels(pcd_np[:,3])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        pos=torch_points
                        )
            
            # #########################
            # debug visualization
            import networkx as nx
            from torch_geometric.utils.convert import to_networkx
            datagraph = to_networkx(data)

            node_labels = data.y[list(datagraph.nodes)].cpu().numpy()

            import matplotlib.pyplot as plt
            plt.figure(1,figsize=(14,12)) 
            nx.draw(datagraph, cmap=plt.get_cmap('Set1'),node_color = node_labels,node_size=4,linewidths=3)
            # plt.show()
            plt.savefig("40k_graph.png")
            # ##########################
            torch.save(data, 
                        os.path.join(self.processed_dir, 
                        f'data_{index}.pt'))
            # if self.test:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_test_{index}.pt'))
            # else:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_{index}.pt'))
            index +=1

    def _get_labels(self, label):
        # label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float16)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """

        data = torch.load(os.path.join(self.processed_dir, 
                        f'data_{idx}.pt'))  
        # if self.test:
        #     data = torch.load(os.path.join(self.processed_dir, 
        #                          f'data_test_{idx}.pt'))
        # else:
        #     data = torch.load(os.path.join(self.processed_dir, 
        #                          f'data_{idx}.pt'))   
        return data


def preprocess_point_cloud(pcd, sampling_rate, normal_radius):
    print(":: Downsample keeping every  %.3f-th point." % sampling_rate)
    pcd_down = pcd.voxel_down_sample(0.0001)
    # pcd_down = pcd.uniform_down_sample(sampling_rate)

    normal_radius = 0.1
    print(":: Estimate normal with search radius %.3f." % normal_radius)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    o3d.io.write_point_cloud("SUNRGBDtoolbox/pcd_data_testing/3_sampled_norm_pcd/2_test_voxel_sample_0.1_norm.pcd" , pcd_down)
    # radius_feature = voxel_size * 10
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     pcd_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pcd_fpfh = None
    return pcd_down, pcd_fpfh


def main():
    # ########################################## #
    # all parameters are for metric point clouds #
    # ########################################## #
    # pcd_name = 'SUNRGBDtoolbox/pcd_data_testing_2/raw/sunrgbd_xyzrgbi_1.txt'
    # pcd_name = 'SUNRGBDtoolbox/pcd_data_testing_2/raw/sunrgbd_xyzrgbi_7.txt'
    # pcd_object = PyntCloud.from_file("gnn_dataset/cpp_output_demo_dir/sunrgbd_pcd_1.pcd")
    
    # pcd_np = np.genfromtxt(pcd_name, delimiter=",")

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])
    # pcd.colors = o3d.utility.Vector3dVector(pcd_np[:,3:6]/255)
    
    
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    # vox_normal_pcd, fpfh_pcd = preprocess_point_cloud(pcd,sampling_rate=3,normal_radius=0.1)
    # points = np.asarray(pcd.points)
    # torch_points = torch.tensor(points)
    # o3d.visualization.draw_geometries([vox_normal_pcd])
    # o3d.io.write_point_cloud("SUNRGBDtoolbox/pcd_data_testing/sampled_pcd/orig.pcd" , pcd)

    # dataset = MyOwnDataset(root='')
    dataset = MyOwnDataset(root='gnn_dataset/pyg_data/')

    print(dataset[0].edge_index.t())
    print(dataset[0].x)
    print(dataset[0].y)
    print(dataset[0].pos)

if __name__=='__main__':
    main()