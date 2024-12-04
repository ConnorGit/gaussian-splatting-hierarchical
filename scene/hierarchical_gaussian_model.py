import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scene.gaussian_model import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class HierarchicalGaussianModel(GaussianModel):

    def __init__(self, sh_degree, optimizer_type="default"):
        GaussianModel.__init__(self, sh_degree, optimizer_type)
        self._centriod_xyz = torch.empty(0) # list of k cords 
        self._centriod_scaling = torch.empty(0) # list of k scales 
        self._centriod_rotation = torch.empty(0) # list of k rotations 
        self._centriod_matching = torch.empty(0) # list of query vectors kxd
        self._patch_matching = torch.empty(0) # key vector 1xd
        
        self.patches = [] # num patches P
        
        self._flattened_xyz = torch.empty(0)
        self._flattened_features_dc = torch.empty(0)
        self._flattened_features_rest = torch.empty(0)
        self._flattened_opacity = torch.empty(0)
        self._flattened_covariance = torch.empty(0)
        
        self.dropout_p=0.0
        self.samples_per_centriod = 1
        self.depth = None
        self.scaling_mod = 1.0
    
    def attention(self):
        scale_factor = 1 / math.sqrt(self._centriod_matching.size(-1))
        keys = torch.stack([patch._patch_matching for patch in self.patches])
        attn_weight = self._centriod_matching @ keys.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, train=True)
        return attn_weight

    def flatten(self): 
        all_xyz = []
        all_features_dc = []
        all_features_reset = []
        all_covariance = []
        all_opacity = []

        all_xyz.append(self.xyz)
        all_covariance.append(self.covariance_activation(super().get_scaling, self.scaling_mod, self._rotation))
        all_features_dc.append(super().get_features_dc)
        all_features_reset.append(super().get_features_rest)
        all_opacity.append(self._opacity)

        attn_weight = self.attention()

        for c in range(self._centriod_xyz.size(0)):
            indices = torch.multinomial(attn_weight[c], self.samples_per_centriod, replacement=False)

            for i in indices:
                patch=self.patches[i]
                centriod_transform = build_scaling_rotation(self.get_centriod_scaling[c], self._centriod_rotation[c])
                centroid_trasformed_xyz = self._centriod_xyz[c] + torch.matmul(centriod_transform, patch._flattened_xyz)

                flattened_transform = build_scaling_rotation(patch.get_centriod_scaling * self.scaling_mod, patch._centriod_rotation)
                new_covariance = flattened_transform @ flattened_transform.transpose(1, 2)
                symm = strip_symmetric(new_covariance)

                all_xyz.append(centroid_trasformed_xyz)
                all_covariance.append(symm)
                all_features_dc.append(patch._flattened_features_dc)
                all_features_reset.append(patch._flattened_features_rest)
                all_opacity.append(attn_weight[c, i] * patch._flattened_opacity)

        self._flattened.xyz = torch.stack(all_xyz)
        self._flattened_features_dc = torch.stack(all_features_dc)
        self._flattened_features_rest = torch.stack(all_features_reset)
        self._flattened_covariance = torch.stack(all_covariance)
        self._flattened_opacity = torch.stack(all_opacity)

    @property
    def get_scaling(self):
        raise Exception('tried to call hierarchical get_scaling')
    
    @property
    def get_rotation(self):
        raise Exception('tried to call hierarchical get_rotation')
    
    @property
    def get_xyz(self):
        return self._flattened_xyz
    
    @property
    def get_features(self):
        features_dc = self._flattened_features_dc
        features_rest = self._flattened_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._flattened_features_dc
    
    @property
    def get_features_rest(self):
        return self._flattened_features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._flattened_opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self._flattened_covariance