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

import numpy as np
import glob
import imageio
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import torchvision

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def subPixels(img, xs, ys):
    height = img.shape[0]
    width = img.shape[1]
    xs = np.reshape(xs, -1)
    ys = np.reshape(ys, -1)
    ix0 = xs.astype(int)
    iy0 = ys.astype(int)
    ix1 = ix0+1
    iy1 = iy0+1

    ids = np.reshape(np.where(ix0 < 0), -1)
    if len(ids) > 0:
        ix0[ids]=0
        ix1[ids]=0
    ids = np.reshape(np.where(iy0 < 0), -1)
    if len(ids) > 0:
        iy0[ids] = 0
        iy1[ids] = 0
    ids = np.reshape(np.where(ix1 >= width-1), -1)
    if len(ids) > 0:
        ix0[ids] = width-1
        ix1[ids] = width-1
    ids = np.reshape(np.where(iy1 >= height - 1), -1)
    if len(ids) > 0:
        iy0[ids] = height - 1
        iy1[ids] = height - 1


    ratex = xs - ix0
    ratey = ys - iy0
    if len(img.shape) > 2:
        ratex = ratex.reshape((-1,1))
        ratey = ratey.reshape((-1, 1))

    px0_y0 = img[(iy0,ix0)]
    px0_y1 = img[(iy1,ix0)]
    px1_y0 = img[(iy0,ix1)]
    px1_y1 = img[(iy1,ix1)]

    py0 = px0_y0 * (1.0-ratex) + px1_y0*ratex
    py1 = px0_y1 * (1.0-ratex) + px1_y1*ratex
    p = py0 * (1 - ratey) + py1 * ratey
    return p

def rotateVector(vector, axis, angle):
    cos_ang = np.reshape(np.cos(angle),(-1));
    sin_ang = np.reshape(np.sin(angle),(-1));
    vector = np.reshape(vector,(-1,3))
    axis = np.reshape(np.array(axis),(-1,3))
    return vector * cos_ang[:,np.newaxis] + axis*np.dot(vector,np.transpose(axis))*(1-cos_ang)[:,np.newaxis] + np.cross(axis,vector) * sin_ang[:,np.newaxis]


#Phi Theta
def SphToVec(coords):
    coords = np.reshape(coords,(-1,2))

    vec = np.zeros((coords.shape[0],3))
    vec[:,0] = np.cos(coords[:,0])*np.sin(coords[:,1])
    vec[:,1] = np.sin(coords[:,0])*np.sin(coords[:,1])
    vec[:,2] = np.cos(coords[:,1])
    return vec

def compute_dist(w, h, fov=57):
    """
    fov: in degree
    """
    fov = fov * np.pi / 180
    return 0.5 * w / np.tan(0.5 * fov)

def get_rays(H,W,K,c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
    


def get_rays_stanford(H, W, focal, cam_center,R,imgscale=1):
    i, j = np.meshgrid(np.arange(W*imgscale, dtype=np.float32), np.arange(H*imgscale, dtype=np.float32), indexing='xy')

    i = i[::imgscale,::imgscale]
    j = j[::imgscale,::imgscale]

    dirs = np.stack([(i-W*imgscale*.5)/focal, (j-H*imgscale*.5)/focal, np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * R, -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # rays_d = dirs
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(cam_center, np.shape(rays_d))
    return rays_o, rays_d


def rayPlaneInter(n,p0,ray_o,ray_d):
    
    s1 = np.sum(p0 * n,1)
    s2 = np.sum(ray_o * n,1)

    s3 = np.sum(ray_d * n,1)

    dist = (s1 - s2) /s3

    # dist_group = np.tile(np.expand_dims(dist,axis=2),(1,1,3))
    dist_group = np.broadcast_to(dist,(3,dist.shape[0])).T


    inter_point = ray_o + dist_group * ray_d

    return inter_point






