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
    
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))