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

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




class Nerf4D_relu_ps(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=4, skips=[4,8,12,16,20],depth_branch=False):
        """ 
        """
        super(Nerf4D_relu_ps, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch

        self.skips = np.arange(4, D, 4)

        # self.pe_flag = pe
        # self.skips = skips
        # self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.L=8
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(self.L*8 + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])

        # if use_viewdirs:
        self.feature_linear = nn.Linear(W, W)

        self.depth_linear = nn.Linear(W, W//2)
        self.depth_linear2 = nn.Linear(W//2, 1)
        
        self.rgb_linear = nn.Linear(W//2, 3)

        self.rgb_act   = nn.Sigmoid()
        self.depth_act = nn.Sigmoid() 

        self.input_net =  nn.Linear(4, input_ch)
       
        self.b = Parameter(torch.normal(mean=0.0, std=3, size=(int(input_ch/2), 4)), requires_grad=False)

        self.input_net_pe =  nn.Linear(self.L*8, input_ch)

        self.depth_flag = depth_branch

    def forward(self, x):
        
        input_pts = self.input_net(x)

        input_pts = F.relu(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)

        feature = self.feature_linear(h)
        h = feature

        if(self.depth_flag):
            depth = self.depth_linear(h)
            depth = F.relu(depth)
            depth = self.depth_linear2(depth)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        if(self.depth_flag):
            depth = self.depth_act(depth)

        rgb   = self.rgb_act(rgb)

        if(self.depth_flag):
            return rgb,depth
        else:
            return rgb
