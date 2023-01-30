# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import imageio
import numpy as np
import argparse
import os
import sys
cpwd = os.getcwd()
sys.path.append(cpwd)
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R_func
from tqdm import tqdm
from src.load_llfff import load_llff_data
from src.cam_view import rayPlaneInter
from src.utils import get_rays_np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default = 'dataset/Ollie/',help = 'exp name') #data/llff/nelf/house2/
parser.add_argument("--factor", type=int, default=4, help='downsample factor for LLFF images')

if __name__ == "__main__":
    testskip=60

    args = parser.parse_args()
    data_dir  = args.data_dir
    uvst_path = f"{data_dir}/uvst.npy"
    rgb_path = f"{data_dir}/rgb.npy"
    t_path = f"{data_dir}/trans.npy"
    k_path = f"{data_dir}/k.npy"
    fd_path = f"{data_dir}/fdepth.npy"
    rendpose_path = f"{data_dir}/Render_pose.npy"
    pose_path = f"{data_dir}/cam_pose.npy"
    cam_idx_path   = f"{data_dir}/cam_idx.npy"

    # process the pose
    images, poses, bds, render_poses, i_test,focal_depth = load_llff_data(args.data_dir, args.factor,
                                                                recenter=True, bd_factor=.75)
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape,  hwf, args.data_dir)

    # Cast intrinsics to correct types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]])

    val_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip==0])
    train_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip])
    print(f'val_idx {val_idx} train_idx {train_idx}')
    val_poses = poses[::testskip]
    val_images = images[::testskip]
    train_poses = np.stack([poses[k] for k in range(poses.shape[0]) if k%testskip], axis=0)
    train_images = np.stack([images[k] for k in range(poses.shape[0]) if k%testskip], axis=0)

    ray_o = []
    ray_d = []
    Trans = []
    for p in poses[:,:3,:4]:
        ray_o_unit, ray_d_unit = get_rays_np(H, W, K, p)

        ray_o.append(ray_o_unit)
        ray_d.append(ray_d_unit)
        Trans.append(p[:2,3].T)

    ray_o = np.asarray(ray_o)
    ray_d = np.asarray(ray_d)

    ray_o = np.reshape(ray_o,(-1,3))
    ray_d = np.reshape(ray_d,(-1,3))

    # add view direction vector
    view_dir = []
    view_ray =  ray_o + ray_d
    data_view_dir = -view_ray / np.broadcast_to(np.expand_dims(np.linalg.norm(view_ray,axis=1),1),view_ray.shape)
    view_dir.append(data_view_dir)

    # interset radius plane
    uv_depth = 0.0

    st_depth = -focal_depth

    # interset radius plane 
    def save_idx(idx, label):
        ray_o = []
        ray_d = []
        Trans = []
        cam_idx = []
        cam_poses = []
        for p in poses[idx,:3,:4]:
            ray_o_unit, ray_d_unit = get_rays_np(H, W, K, p)
            ray_o.append(ray_o_unit)
            ray_d.append(ray_d_unit)
            Trans.append(p[:2,3].T)
            cam_idx.append(ray_o_unit)
            cam_poses.append(p[:,3])

        ray_o = np.asarray(ray_o)
        ray_d = np.asarray(ray_d)
        cam_idx = np.asarray(cam_idx)
        cam_poses = np.asarray(cam_poses)

        ray_o = np.reshape(ray_o,(-1,3))
        ray_d = np.reshape(ray_d,(-1,3))
        cam_idx = np.reshape(cam_idx,(-1,3))
        cam_poses = np.reshape(cam_poses,(-1,3))


        plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)
        p_uv = np.broadcast_to(np.array([0.0,0.0,uv_depth]),np.shape(ray_o))
        p_st = np.broadcast_to(np.array([0.0,0.0,st_depth]),np.shape(ray_o))
        inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
        inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)
        data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
        rgb = np.reshape(images[idx],(-1,3))
        trans  = np.asanyarray(Trans)
        intrinsics = K
        
        print("uvst = ", data_uvst.shape)
        print("rgb = ", rgb.shape)
       
        np.save(uvst_path.replace('.npy', f'{label}.npy'), data_uvst)
        np.save(rgb_path.replace('.npy', f'{label}.npy'), rgb)
        np.save(t_path.replace('.npy', f'{label}.npy'), trans)
        np.save(k_path.replace('.npy', f'{label}.npy'), intrinsics)
        np.save(fd_path.replace('.npy', f'{label}.npy'), focal_depth)
        np.save(rendpose_path.replace('.npy', f'{label}.npy'),render_poses[:,:,:4])
    
        np.save(pose_path.replace('.npy', f'{label}.npy'),cam_poses)
        np.save(cam_idx_path.replace('.npy', f'{label}.npy'),cam_idx)

    save_idx(train_idx, 'train')
    save_idx(val_idx, 'val')
