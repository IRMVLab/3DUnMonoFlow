"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
#from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from .point_conv import PointNetSaModule,CostVolume,SetUpconvModule,PointnetFpModule,WarpingLayers,FlowPredictor,Conv1d
from .point_conv import SceneFlowEstimatorPointConv
from .point_conv import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time

scale = 1.0

class HALFlow(nn.Module):
    def __init__(self,flag_big_radius, kernel_shape,layers, is_training, bn_decay=None):
        super(HALFlow, self).__init__()

        RADIUS1 = 0.5
        RADIUS2 = 1.0
        RADIUS3 = 2.0
        RADIUS4 = 4.0

        self.layer0 = PointNetSaModule(flag_big_radius,kernel_shape,layers,npoint=2048, radius=RADIUS1, nsample=32, in_channels=3,mlp=[16,16,32],mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer1 = PointNetSaModule(flag_big_radius, kernel_shape, layers,npoint=1024, radius=RADIUS1, nsample=24, in_channels=32,mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer2 = PointNetSaModule(flag_big_radius, kernel_shape, layers,npoint=256, radius=RADIUS2, nsample=16, in_channels=64,mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer3_2 = PointNetSaModule(flag_big_radius, kernel_shape, layers,npoint=64, radius=RADIUS3, nsample=16,in_channels=128,mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)

        self.cost1 = CostVolume(flag_big_radius, kernel_shape, layers,radius=10.0, nsample=4, nsample_q=32, in_channels=128,mlp1=[256,128,128], mlp2 = [256,128], is_training=is_training, bn_decay=bn_decay, bn=True, pooling='max', knn=True, corr_func='concat')

        self.layer3_1 = PointNetSaModule(flag_big_radius, kernel_shape, layers, npoint=64, radius=RADIUS3, nsample=8,in_channels=128, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer4_1 = PointNetSaModule(flag_big_radius, kernel_shape, layers,npoint=16, radius=RADIUS4, nsample=8,in_channels=256, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)

        self.upconv1 = SetUpconvModule(flag_big_radius, kernel_shape, layers,nsample=8, radius=2.4,in_channels=[256,512],mlp=[256,256,512], mlp2=[512],is_training=is_training, bn_decay=bn_decay, knn=True)


        # Layer3
        self.conv1 = Conv1d(512,3)

        self.warping1 = WarpingLayers()

        self.cost2 = CostVolume(flag_big_radius, kernel_shape, layers,radius=10.0, nsample=4, nsample_q=6, in_channels=256,mlp1=[512,256,256], mlp2=[512,256], is_training=is_training, bn_decay=bn_decay,bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred1 = FlowPredictor([256,512,256], mlp=[512,256,256], is_training = is_training , bn_decay = bn_decay)

        self.conv2 = Conv1d(256,3)


        # Layer 2
        self.upconv2 = SetUpconvModule(flag_big_radius, kernel_shape, layers,nsample=8, radius=1.2, in_channels=[128,256],mlp=[256,128,128], mlp2=[128],is_training=is_training, bn_decay=bn_decay, knn=True)

        self.fp1 = PointnetFpModule(in_channels=3,mlp=[], is_training=is_training, bn_decay=bn_decay)

        self.warping2 = WarpingLayers()

        self.cost3 = CostVolume(flag_big_radius,kernel_shape,layers,radius=10.0, nsample=4, nsample_q=6, in_channels=128,mlp1=[256,128,128], mlp2=[256,128], is_training=is_training, bn_decay=bn_decay, bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred2 = FlowPredictor([128,128,128],mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay)

        self.conv3 = Conv1d(128,3)


        #Layer 1
        self.upconv3 = SetUpconvModule(flag_big_radius, kernel_shape, layers,nsample=8, radius=1.2,in_channels=[64,128], mlp=[256,128,128], mlp2=[128],is_training=is_training, bn_decay=bn_decay, knn=True)

        self.fp2 = PointnetFpModule(in_channels=3,mlp=[],is_training=is_training,bn_decay=bn_decay)

        self.warping3 = WarpingLayers()

        self.cost4 = CostVolume(flag_big_radius, kernel_shape, layers, radius=10.0, nsample=4, nsample_q=6, in_channels=64,mlp1=[128,64,64], mlp2=[128,64],is_training=is_training, bn_decay=bn_decay,bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred3 = FlowPredictor([64,128,64],mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay)

        self.conv4 = Conv1d(128,3)


        self.fp3 = PointnetFpModule(in_channels = 160,mlp=[256,256], is_training=is_training, bn_decay=bn_decay)

        self.conv5 = Conv1d(256,128)

        self.fp4 = PointnetFpModule( in_channels=3,mlp=[], is_training=is_training, bn_decay=bn_decay)

        self.conv6 = Conv1d(128,3)


    def forward(self, xyz1,xyz2,color1,color2,label=None):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3
        # label: B,N,3
        xyz1 = xyz1.permute(0,2,1).contiguous()
        xyz2 = xyz2.permute(0,2,1).contiguous()
        color1 = color1.permute(0,2,1).contiguous()
        color2 = color2.permute(0,2,1).contiguous()

        l0_xyz_f1_raw = xyz1
        l0_xyz_f2_raw = xyz2

        xyz1_center = torch.mean(xyz1,dim=1,keepdim=True)   # (B,1,3)
        xyz1 = xyz1 - xyz1_center   # (B,N,3)
        xyz2 = xyz2 - xyz1_center   # (B,N,3)

        l0_xyz_f1 = xyz1

        l0_points_f1 = color1

        if label is None:
            label = torch.zeros(xyz1.size(),device='cuda')

        l0_label_f1 = label

        l0_xyz_f2 = xyz2

        l0_points_f2 = color2

        # N = 8192
        l0_xyz_f1, l0_label_f1, l0_points_f1,l0_idx1, pc1_sample = self.layer0(l0_xyz_f1, l0_xyz_f1_raw, l0_label_f1, l0_points_f1)  #(B,2048,3) (B,2048,3) (B,2048,32)

        l1_xyz_f1, l1_label, l1_points_f1, l1_idx1 = self.layer1(l0_xyz_f1, None, l0_label_f1, l0_points_f1)  #(B,1024,3) (B,1024,3) (B,1024,64)

        l2_xyz_f1, l2_label, l2_points_f1,  l2_idx1 = self.layer2(l1_xyz_f1, None, l1_label, l1_points_f1)    #(B,256,3) (B,256,3) (B,256,128)


        l0_xyz_f2, _, l0_points_f2, l0_idx2, pc2_sample = self.layer0(l0_xyz_f2, l0_xyz_f2_raw, label, l0_points_f2)  #(B,2048,3) (B,2048,3) (B,2048,32)

        l1_xyz_f2, _, l1_points_f2, l1_idx2  = self.layer1(l0_xyz_f2, None, l0_label_f1, l0_points_f2)  #(B,1024,3) (B,1024,3) (B,1024,64)

        l2_xyz_f2, _, l2_points_f2, l2_idx2 = self.layer2(l1_xyz_f2, None, l1_label, l1_points_f2)   #(B,256,3) (B,256,3) (B,256,128)

        l3_xyz_f2, _, l3_points_f2, l3_idx2 = self.layer3_2(l2_xyz_f2, None, l2_label, l2_points_f2)  #(B,64,3) (B,64,3) (B,64,256)

        l2_points_f1_new = self.cost1(l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2)  # (B,256,128)

        l3_xyz_f1, l3_label, l3_points_f1,l3_idx1 = self.layer3_1(l2_xyz_f1, None, l2_label, l2_points_f1_new) # (B,64,3) (B,64,3) (B,64,256)

        l4_xyz_f1, _, l4_points_f1,l4_idx1 = self.layer4_1(l3_xyz_f1, None, l3_label, l3_points_f1)  #(B,16,3) (B,16,3) (B,16,512)


        l3_feat_f1 = self.upconv1(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1)  #(B,64,512)


        #Layer 3
        l3_points_f1_new = l3_feat_f1  #(B,64,512)

        l3_flow_coarse = self.conv1(l3_points_f1_new)  #(B,64,3)

        l3_flow_warped = self.warping1(l3_xyz_f1, l3_flow_coarse)   #(B,64,3)

        l3_cost_volume = self.cost2(l3_flow_warped, l3_points_f1, l3_xyz_f2, l3_points_f2)  #(B,64,256)

        l3_flow_finer = self.flow_pred1(l3_points_f1, l3_points_f1_new, l3_cost_volume)  # (B,64,256)

        l3_flow_det = self.conv2(l3_flow_finer)

        l3_flow = l3_flow_coarse + l3_flow_det  #(B,64,3)

        #Layer 2
        l2_points_f1_new = self.upconv2(l2_xyz_f1, l3_xyz_f1, l2_points_f1, l3_flow_finer)  #(B,256,128)

        l2_flow_coarse = self.fp1(l2_xyz_f1, l3_xyz_f1, None, l3_flow)  #(B,256,3)

        l2_flow_warped = self.warping2(l2_xyz_f1, l2_flow_coarse)  #(B,256,3)

        l2_cost_volume = self.cost3(l2_flow_warped, l2_points_f1, l2_xyz_f2, l2_points_f2)  #(B,256,128)

        l2_flow_finer = self.flow_pred2(l2_points_f1, l2_points_f1_new, l2_cost_volume)   #(B,256,128)

        l2_flow_det = self.conv3(l2_flow_finer)    #(B,256,3)

        l2_flow = l2_flow_coarse + l2_flow_det    #(B,256,3)

        #Layer 1
        l1_points_f1_new = self.upconv3(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_flow_finer)   #(B,1024,128)

        l1_flow_coarse = self.fp2(l1_xyz_f1, l2_xyz_f1, None, l2_flow)  #(B,1024,3)

        l1_flow_warped = self.warping3(l1_xyz_f1, l1_flow_coarse)  #(B,1024,3)

        l1_cost_volume = self.cost4(l1_flow_warped, l1_points_f1, l1_xyz_f2, l1_points_f2)    #(B,1024,64)

        l1_flow_finer = self.flow_pred3(l1_points_f1, l1_points_f1_new, l1_cost_volume)   #(B,1024,128)

        l1_flow_det = self.conv4(l1_flow_finer)     #(B,1024,3)

        l1_flow = l1_flow_coarse + l1_flow_det   #(B,1024,3)

        l0_feat_f1 = self.fp3(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_flow_finer)   #(B,2048,256)

        net = self.conv5(l0_feat_f1)   #(B,2048,128)

        l0_flow_coarse = self.fp4(l0_xyz_f1, l1_xyz_f1, None, l1_flow)   #(B,2048,3)

        l0_flow_det = self.conv6(net)    #(B,2048,3)

        l0_flow = l0_flow_coarse + l0_flow_det    #(B,2048,3)

        l0_flow = l0_flow.permute(0,2,1)
        l1_flow = l1_flow.permute(0,2,1)
        l2_flow = l2_flow.permute(0,2,1)
        l3_flow = l3_flow.permute(0,2,1)
        l0_label_f1 = l0_label_f1.permute(0,2,1)
        l1_label = l1_label.permute(0,2,1)
        l2_label = l2_label.permute(0,2,1)
        l3_label = l3_label.permute(0,2,1)
        l0_xyz_f1 = l0_xyz_f1.permute(0,2,1)
        l1_xyz_f1 = l1_xyz_f1.permute(0,2,1)
        l2_xyz_f1 = l2_xyz_f1.permute(0,2,1)
        l3_xyz_f1 = l3_xyz_f1.permute(0,2,1)
        l0_xyz_f2 = l0_xyz_f2.permute(0, 2, 1)
        l1_xyz_f2 = l1_xyz_f2.permute(0, 2, 1)
        l2_xyz_f2 = l2_xyz_f2.permute(0, 2, 1)
        l3_xyz_f2 = l3_xyz_f2.permute(0, 2, 1)
        flow = [l0_flow, l1_flow, l2_flow, l3_flow]
        label = [l0_label_f1, l1_label, l2_label, l3_label]
        pc1 = [l0_xyz_f1, l1_xyz_f1, l2_xyz_f1, l3_xyz_f1]
        pc2 = [l0_xyz_f2, l1_xyz_f2, l2_xyz_f2, l3_xyz_f2]
        idx1 = [l0_idx1, l1_idx1, l2_idx1, l3_idx1]
        idx2 = [l0_idx2, l1_idx2, l2_idx2, l3_idx2]

        return flow, idx1, idx2


def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm  # B N 5

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(cur_pc1, cur_pc2, cur_flow):
    
    #compute curvature
    cur_pc2_curvature = curvature(cur_pc2)

    cur_pc1_warp = cur_pc1 + cur_flow
    dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
    moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

    chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

    #smoothness
    smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

    #curvature
    inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
    curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()


    return chamferLoss, curvatureLoss, smoothnessLoss

def interpolateFea(pc1, pc2, fea2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    fea2: B M 3
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False) # B N 5
    grouped_fea2 = index_points_group(fea2, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm  # B N 5
    inter_fea2 = torch.sum(weight.view(B, N, 5, 1) * grouped_fea2, dim = 2)
    return inter_fea2 # B N 3

def rgbConsistencyLoss(pc1,pc2,fea1,fea2,flow):
    fea1 = fea1.permute(0,2,1)
    fea2 = fea2.permute(0,2,1)
    pred_pc2 = pc1 + flow  #B C N
    inter_pc2_fea = interpolateFea(pred_pc2,pc2,fea2)
    rgbLoss = (inter_pc2_fea - fea1).abs().mean()
    return rgbLoss


if __name__ == "__main__":

    num_points = 8192
    xyz1 = torch.rand(1, num_points, 3).cuda()
    xyz2 = torch.rand(1, num_points, 3).cuda()
    color1 = torch.rand(1, num_points, 3).cuda()
    color2 = torch.rand(1, num_points, 3).cuda()

    gt_flow = torch.rand(1, num_points, 3).cuda()
    model = HALFlow().cuda()

    model.eval()
    for _ in range(1):
        with torch.no_grad():
            flows, idx1, idx2 = model(xyz1, xyz2, color1, color2)
            torch.cuda.synchronize()


    self_loss = multiScaleChamferSmoothCurvature(xyz1[idx1[0]],xyz2[idx2[0]], flows[0])

    print(flows[0].shape, self_loss)
