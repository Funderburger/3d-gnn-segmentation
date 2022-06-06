
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import BaseTransform, Compose, FaceToEdge, KNNGraph, RadiusGraph
from torch_geometric.data import Data, InMemoryDataset, extract_zip, Dataset, download_url
from torch_geometric.loader import DataLoader
import torch.distributed as dist

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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

import os
import os.path as osp

from pyntcloud import PyntCloud

import scipy.io
from zmq import device
# mat = scipy.io.loadmat('SUNRGBDtoolbox/Metadata/seglistall.mat')
# mat['seglistall'][0][5]
# torch.cuda.set_device(1)



torch.cuda.empty_cache()

class NormalizeUnitSphere(BaseTransform):
    """Center and normalize node-level features to unit length."""

    @staticmethod
    def _re_center(x):
        """Recenter node-level features onto feature centroid."""
        centroid = torch.mean(x, dim=0)
        return x - centroid

    @staticmethod
    def _re_scale_to_unit_length(x):
        """Rescale node-level features to unit-length."""
        max_dist = torch.max(torch.norm(x, dim=1))
        return x / max_dist

    def __call__(self, data: Data):
        if data.x is not None:
            data.x = self._re_scale_to_unit_length(self._re_center(data.x))

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MyOwnDataset(Dataset):
    def __init__(self, validation, root, transform=None, pre_transform=None, pre_filter=None):
        self.validation = validation
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
        validation_size = int(len(self.data)*0.3)

        # return [f'data_{i}.pt' for i in range(len(self.data))]
        if self.validation:
            return [f'val_data_{i}.pt' for i in range(validation_size)]
        else:
            return [f'data_{i}.pt' for i in range(len(self.data)-validation_size)]

    def download(self):
        pass

    def process(self):
        # idx = 0
        # files = []
        # for filename in sorted(os.scandir(self.raw_paths[0]), key=lambda f: f.name):
        #     files.append(filename)
        
        self.data = self.raw_paths
        validation_size = int(len(self.data)*0.3)
        validation_data = self.data[-validation_size:]
        training_data = self.data[:len(self.data)-validation_size]
        index = 0
        if self.validation:
            dataset = validation_data
        else:
            dataset = training_data

        for pcd_name in tqdm(dataset, total=dataset.__len__()):
            # pcd_np = np.genfromtxt(pcd_name, delimiter=",")
            pcd_object = PyntCloud.from_file(pcd_name)
            pcd_np = np.array([pcd_object.points.label, pcd_object.points.x, pcd_object.points.y, pcd_object.points.z, pcd_object.points.rgba]).T

            # Get node features
            # node_feats = torch.tensor(np.concatenate([pcd_np[:,:3], pcd_np[:,3:6]/255],1)).cuda()
            node_feats = torch.tensor(pcd_np[:,1:5]).cuda()
            # Get edge features
            edge_feats = None
            # Get adjacency info
            torch_points = torch.tensor(pcd_np[:,1:4]).cuda()
            edge_index = torch_geometric.nn.knn_graph(
                torch_points,
                4,
                None,
                loop=False,
                flow='source_to_target',
                cosine=True,
                num_workers=8,
            )
            # Get labels info
            # label = self._get_labels(pcd_np[:,6])
            label = self._get_labels(pcd_np[:,0])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        pos=torch_points
                        )
            

            # ****************************
            # * debug visualization
            # ****************************
            # import networkx as nx
            # from torch_geometric.utils.convert import to_networkx
            # datagraph = to_networkx(data)

            # node_labels = data.y[list(datagraph.nodes)].cpu().numpy()

            # import matplotlib.pyplot as plt
            # plt.figure(1,figsize=(14,12)) 
            # nx.draw(datagraph, cmap=plt.get_cmap('Set1'),node_color = node_labels,node_size=4,linewidths=3)
            # # plt.show()
            # plt.savefig(pcd_name[-6:-3]+"40k_graph.png")
            # # ##########################
            # ***********************************
            # ***********************************
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            if self.validation:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'val_data_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
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
    
        if self.validation:
            data = torch.load(os.path.join(self.processed_dir, 
                                f'val_data_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                f'data_{idx}.pt')) 
        return data


class FeatureSteeredConvolution(MessagePassing):
    """Implementation of feature steered convolutions.

    References
    ----------
    .. [1] Verma, Nitika, Edmond Boyer, and Jakob Verbeek.
       "Feastnet: Feature-steered graph convolutions for 3d shape analysis."
       Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super().__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops

        self.linear = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels * num_heads,
            bias=False,
        )
        self.u = torch.nn.Linear(
            in_features=in_channels,
            out_features=num_heads,
            bias=False,
        )
        self.c = torch.nn.Parameter(torch.Tensor(num_heads))

        if not ensure_trans_invar:
            self.v = torch.nn.Linear(
                in_features=in_channels,
                out_features=num_heads,
                bias=False,
            )
        else:
            self.register_parameter("v", None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of tuneable network parameters."""
        torch.nn.init.uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.u.weight)
        torch.nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)
        if self.v is not None:
            torch.nn.init.uniform_(self.v.weight)

    def forward(self, x, edge_index):
        """Forward pass through a feature steered convolution layer.

        Parameters
        ----------
        x: torch.tensor [|V|, in_features]
            Input feature matrix, where each row describes
            the input feature descriptor of a node in the graph.
        edge_index: torch.tensor [2, E]
            Edge matrix capturing the graph's
            edge structure, where each row describes an edge
            between two nodes in the graph.
        Returns
        -------
        torch.tensor [|V|, out_features]
            Output feature matrix, where each row corresponds
            to the updated feature descriptor of a node in the graph.
        """
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.shape[0])

        out = self.propagate(edge_index, x=x)
        return out if self.bias is None else out + self.bias

    def _compute_attention_weights(self, x_i, x_j):
        """Computation of attention weights.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, M]
            Matrix of attention scores, where each row captures
            the attention weights of transformed node in the graph.
        """
        if x_j.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected input features with {self.in_channels} channels."
                f" Instead received features with {x_j.shape[-1]} channels."
            )
        if self.v is None:
            attention_logits = self.u(x_i - x_j) + self.c
        else:
            attention_logits = self.u(x_i) + self.b(x_j) + self.c
        return F.softmax(attention_logits, dim=1)

    def message(self, x_i, x_j):
        """Message computation for all nodes in the graph.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, out_features]
            Matrix of updated feature embeddings for
            all nodes in the graph.
        """
        attention_weights = self._compute_attention_weights(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attention_weights.view(-1, self.num_heads, 1) * x_j).sum(dim=1)

class GraphFeatureEncoder(torch.nn.Module):
    """Graph neural network consisting of stacked graph convolutions."""
    def __init__(
        self,
        in_features,
        conv_channels,
        num_heads,
        apply_batch_norm: int = True,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super().__init__()

        conv_params = dict(
            num_heads=num_heads,
            ensure_trans_invar=ensure_trans_invar,
            bias=bias,
            with_self_loops=with_self_loops,
        )
        self.apply_batch_norm = apply_batch_norm

        *first_conv_channels, final_conv_channel = conv_channels
        conv_layers = get_conv_layers(
            channels=[in_features] + conv_channels,
            conv=FeatureSteeredConvolution,
            conv_params=conv_params,
        )
        self.conv_layers = nn.ModuleList(conv_layers)

        self.batch_layers = [None for _ in first_conv_channels]
        if apply_batch_norm:
            self.batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(channel) for channel in first_conv_channels]
            )

    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers
        for conv_layer, batch_layer in zip(first_conv_layers, self.batch_layers):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            if batch_layer is not None:
                x = batch_layer(x)
        return final_conv_layer(x, edge_index)

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

class MeshSeg(torch.nn.Module):
    """Mesh segmentation network."""
    def __init__(
        self,
        in_features,
        encoder_features,
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        num_heads,
        apply_batch_norm=True,
    ):
        super().__init__()
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
        )
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        torch.autograd.set_detect_anomaly(True)
        x, edge_index = data.x, data.edge_index #.to(torch.cuda.device(self).idx), data.edge_index.to(torch.cuda.device(self).idx)
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)


def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_conv_layers(channels: list, conv: MessagePassing, conv_params: dict):
    """Define convolution layers with specified in and out channels.

    Parameters
    ----------
    channels: list
        List of integers specifying the size of the convolution channels.
    conv: MessagePassing
        Convolution layer.
    conv_params: dict
        Dictionary specifying convolution parameters.

    Returns
    -------
    list
        List of convolutions with the specified channels.
    """
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers

def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        # loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss = loss_fn(out, data.y.type(torch.LongTensor).to(device))
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)

def accuracy(predictions, gt_seg_labels):
    """Compute accuracy of predicted segmentation labels.

    Parameters
    ----------
    predictions: [|V|, num_classes]
        Soft predictions of segmentation labels.
    gt_seg_labels: [|V|]
        Ground truth segmentations labels.
    Returns
    -------
    float
        Accuracy of predicted segmentation labels.    
    """
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return float(correct_assignments / num_assignemnts)


def evaluate_performance(dataset, net, device):
    """Evaluate network performance on given dataset.

    Parameters
    ----------
    dataset: DataLoader
        Dataset on which the network is evaluated on.
    net: torch.nn.Module
        Trained network.
    device: str
        Device on which the network is located.

    Returns
    -------
    float:
        Mean accuracy of the network's prediction on
        the provided dataset.
    """
    prediction_accuracies = []
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        gt_labels = data.y.type(torch.LongTensor).to(device)
        gt_labels = gt_labels[:,None]
        prediction_accuracies.append(accuracy(predictions, gt_labels))
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = evaluate_performance(train_data, net, device)
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, test_acc

def load_model(model_params, path_to_checkpoint, device):
    try:
        model = MeshSeg(**model_params)
        model.load_state_dict(
            torch.load(str(path_to_checkpoint)),
            strict=True,
        )
        model.to(device)
        return model
    except RuntimeError as err_msg:
        raise ValueError(
            f"Given checkpoint {str(path_to_checkpoint)} could"
            f" not be loaded. {err_msg}"
        )


def get_best_model(model_params, dataset, device):
    # Just if you want to compare two models
    path_to_pretrained_model = Path("gnn_results/checkpoints/best_pretrained_checkpoint")
    path_to_trained_model = Path("gnn_results/checkpoints/best_checkpoint_model")

    pretrained_model = load_model(
        model_params,
        path_to_pretrained_model,
        device,
    )
    if not path_to_trained_model.exists():
        return pretrained_model
    
    trained_model = load_model(
        model_params,
        path_to_trained_model,
        device,
    )
    acc_pretrained = evaluate_performance(dataset, pretrained_model, device) 
    acc_trained = evaluate_performance(dataset, trained_model, device)
    if acc_pretrained > acc_trained:
        return pretrained_model
    return trained_model


def main(rank, world_size):
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

    # ***************
    # ***************

    # try:
    #     torch.multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # device = 'cpu'
    model_params = dict(
    in_features=4,
    encoder_features=16,
    conv_channels=[32, 64, 128, 64],
    encoder_channels=[16],
    decoder_channels=[32],
    num_classes=38, # 37 different classes + 1 class which is for anything else for the SUNRGBD dataset, of course
    num_heads=38, 
    apply_batch_norm=True,
    )

    # net = MeshSeg(**model_params).double()
    # # net = net.double()
    # # net = net.to(device)
    # # rank = 0
    # # world_size = 1
    # # setup(rank, world_size)
    # # dist.init_process_group("gloo",rank=0, world_size=1)
    # # multi_net = torch.nn.DataParallel(net).to(device) #, device_ids=None) #.to(rank), device_ids=[rank])
    # multi_net = net.to(device)
    # multi_net = net
    device= rank

    net = MeshSeg(**model_params).double().to(rank)
    multi_net = DDP(net, device_ids=[rank])

    root = "gnn_dataset/pyg_data_xyz/"
    pre_transform = Compose([NormalizeUnitSphere()])

    # dataset = MyOwnDataset(validation=True, root='gnn_dataset/pyg_data/')

    # print(dataset[0].edge_index.t())
    # print(dataset[0].x)
    # print(dataset[0].y)
    # print(dataset[0].pos)
    train_data = MyOwnDataset(
        root=root,
        validation= False,
        pre_transform=pre_transform,
    )
    
    validation_data = MyOwnDataset(
        root=root,
        validation=True,
        pre_transform=pre_transform,
    )

    train_loader = DataLoader(train_data,  shuffle=True, batch_size=1) #, num_workers=1, batch_size=1)
    validation_loader = DataLoader(validation_data, shuffle=False, batch_size=1) #, num_workers=1, batch_size=1)

    lr = 0.001
    num_epochs = 200
    best_test_acc = 0.0

    optimizer = torch.optim.Adam(multi_net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()


    with tqdm(range(num_epochs), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(multi_net, train_loader, optimizer, loss_fn, device)
            train_acc, test_acc = test(multi_net, train_loader, validation_loader, device)
            
            tepochs.set_postfix(
                train_loss=train_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
            )
            sleep(0.1)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(multi_net.state_dict(), "gnn_results/checkpoints_xyz/best_checkpoint_model")

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 3
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    # main()