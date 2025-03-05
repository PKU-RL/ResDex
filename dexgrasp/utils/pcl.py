import numpy as np
import os, sys
import trimesh
import queue
import time
import torch
from queue import Empty

def get_pointcloud_from_mesh(mesh_dir, filename, num_sample=4096):
    all_points = []
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename))
    points = mesh.sample(num_sample)
    return points


########## torch point clouds processing ##########

def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def load_object_point_clouds(object_files, asset_root):
    ret = []
    for fn in object_files:
        substrs = fn.split('/')
        pc_fn = os.path.join(substrs[0], 'pointclouds', substrs[-1].replace('.urdf','.npy'))
        print("object file: {} -> pcl file: {}".format(fn, pc_fn))
        pc = np.load(os.path.join(asset_root, pc_fn))
        #pc = np.load("vision/real_pcl.npy")
        ret.append(pc)
    return ret

# if __name__=="__main__":
#     #pc = get_pointcloud_from_mesh('../../assets/obj/meshes', 'cup.obj')
#     #print(pc)

#     #vis_pointcloud(pc)
#     pc = np.load('real_pcl.npy')  #('../../assets/ycb_assets/pointclouds/014_lemon.npy')
#     #pc = np.load('../../assets/obj/pointclouds/cup.npy')
#     print(pc.shape, pc.dtype)

#     # pc = torch.tensor(pc, dtype=torch.float32).to('cuda').unsqueeze(0).expand(512,-1,-1)
#     # print(pc.shape)
#     # from tqdm import trange
#     # for i in trange(1):
#     #     idx = farthest_point_sample(pc, 512, 'cuda')
#     #     pc_sample = index_points(pc, idx, 'cuda')
#     #     print(pc_sample.shape)

    
#     pc_th = torch.tensor(pc, dtype=torch.float32).to('cuda').unsqueeze(0)
#     Q = queue.Queue(1)
#     vis_pointcloud_realtime(Q, coord_len=0.1)
#     for t in range(1000):
#         indices = farthest_point_sample(pc_th, 512, 'cuda')
#         indices = indices[0].cpu().numpy()
#         p = pc[indices]
#         #if Q.full():
#         #    Q.get()
#         Q.put(p)
#         #time.sleep(0.01)
    