# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.sparse import coo_matrix
from .external_util import set_by_index
pixel_coords = None
import numpy as np
import torch.nn.functional as F


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) != h or pixel_coords.size(3) != w:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combination of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combination of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)

def cam2pixel3(cam_coords_flatten, sf, h, w, index):
    b, _, num = cam_coords_flatten.size()
    pcoords = cam_coords_flatten + sf 
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(X).expand(b, h, w)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(Y).expand(b, h, w)  # [bs, H, W]
    
    grid_x = 2*(grid_x/(w-1.0) - 0.5)
    grid_y = 2*(grid_y/(h-1.0) - 0.5)
    
    grid_x_flat = grid_x.view(b, -1)
    grid_y_flat = grid_y.view(b, -1)

    grid_x_flat = set_by_index(grid_x_flat, X_norm, index)
    grid_y_flat = set_by_index(grid_y_flat, Y_norm, index)
    grid_x = grid_x_flat.view(b, h, w)
    grid_y = grid_y_flat.view(b, h, w)

    grid_tf = torch.stack((grid_x,grid_y), dim=3)
    

    return grid_tf, Z.reshape(b, 1, -1)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def flow_warp(img, flow, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        flow: flow map of the target image -- [B, 2, H, W]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'BCHW')
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    grid_tf = torch.stack((X,Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

    return img_tf


def pose2flow(depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode=None):
    """
    Converts pose parameters to rigid optical flow
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)

def pose2scene(depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode=None):
    """
    Converts pose parameters to rigid scene flow
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)


def flow2oob(flow):
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    oob = (X.abs()>1).add(Y.abs()>1)>0
    return oob

def occlusion_mask(grid, depth):
    check_sizes(img, 'grid', 'BHW2')
    check_sizes(depth, 'depth', 'BHW')

    mask = grid

    return mask



def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    return projected_img


def cam2cam2(cam_coords, pose):
    batch, _, height, width = cam_coords.size()
    ones = Variable(torch.ones([batch, 1, height, width])).type_as(cam_coords)
    cam_coords = torch.cat([cam_coords, ones], dim=1).view(batch, 4, height*width)
    cam_coords_T = pose.bmm(cam_coords)
    cam_coords_T=cam_coords_T.view(batch, 4, height, width)

    return cam_coords_T[:, :3, :, :]



def cam2cam(pc_cam1, pose_mat):
    # pose_mat : from cam1 to cam2
    # input the pc in cam1 
    # output the pc in cam2
    # pc_cam1 size(b, n, 3)

    b,_,_ = pose_mat.size()
    rotation = pose_mat[:,:,:3]
    translation = pose_mat[:,:,-1].unsqueeze(-1)

    pc_cam1 = pc_cam1.permute(0,2,1)    #(b,3,n)
    pc_cam2 = rotation.bmm(pc_cam1) + translation
    pc_cam2 = pc_cam2.permute(0,2,1)   #(b,n,3)

    return pc_cam2

def inverse_warp_mask(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', reverse_pose=False, maskp01=False, maskp01_duoci=False, maskp01_qian=None):

    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
   
    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat_revised(pose, rotation_mode)  # [B,3,4]
    if reverse_pose:
        pose_mat = torch.inverse(pose_mat)
    # Get projection matrix for tgt camera frame to source pixel frame
    ones = Variable(torch.zeros([batch_size, 3, 1])).type_as(intrinsics)

    intrinsics = torch.cat([intrinsics, ones], dim=2)
    filler = Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])).type_as(intrinsics).expand(batch_size, 1, 4)
    intrinsics = torch.cat([intrinsics, filler], dim=1)# [B, 4, 4]

    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    if maskp01:
        cam_coords_T = cam2cam2(cam_coords, pose_mat)
        mask_p, mask0, mask1 = mask_p01(depth, src_pixel_coords, cam_coords_T)
        return projected_img, mask_p, mask0, mask1
    if maskp01_duoci:
        mask1 = mask_p01_duoci(src_pixel_coords, maskp01_qian)

        return mask1
    else:
        return projected_img


def mask_p01(depth, coords, cam_coords_T):
      b, h, w, _ = coords.size()

      x0 = torch.floor((w - 1) * (coords[:, :, :, 0] / 2.0 + 0.5).float()).float().cuda()
      x1 = (x0 + 1).float().cuda()
      y0 = torch.floor((h - 1) * (coords[:, :, :, 1] / 2.0 + 0.5).float()).float().cuda()
      y1 = (y0 + 1).float().cuda()


      mask_p_luoji = (x0 >= torch.zeros_like(x0).float().cuda())*(x1 <= torch.Tensor([depth.size()[2] - 1]).float().cuda())*(y0 >= torch.zeros_like(x0).float().cuda())*(y1 <= torch.Tensor([depth.size()[1] - 1]).float().cuda())
      mask_p = mask_p_luoji.float()


      euclidean = torch.sqrt(torch.sum(torch.pow(cam_coords_T, 2), 1)).view(b, -1)
      fuer_like = -2 * torch.ones_like(x0).cuda()
      x0 = torch.where(mask_p_luoji, x0, fuer_like)
      y0 = torch.where(mask_p_luoji, y0, fuer_like)
      xy00 = torch.stack([x0, y0], dim=3).cuda()
      for i in range(b):
        euclideani = euclidean[i, :].view(1, -1)
        xy00_batchi = xy00[i, :, :, :].view(-1, 2).int()  
        unique_xy00_batchi, ids = torch.unique(xy00_batchi[:, 0] * w + xy00_batchi[:, 1],return_inverse=True)

        outputs = coo_matrix((torch.squeeze(1.0 / euclideani, 0).cuda().cpu(), (ids.long().cuda().cpu(), torch.arange(0, euclideani.size()[1] ).long().cuda().cpu())),  shape=(unique_xy00_batchi.size()[0], euclideani.size()[1])).max(1)

        outputs = torch.squeeze(torch.from_numpy(outputs.toarray()).cuda())
        zuixiaojuli = torch.unsqueeze(torch.gather(1.0 / outputs, 0, torch.squeeze(ids).cuda()), 0).float()
        mask0 = torch.unsqueeze(torch.where(zuixiaojuli==euclideani,
                            torch.ones_like(zuixiaojuli).cuda(), torch.zeros_like(zuixiaojuli).cuda()).view(h, w), 0)
        xy00_batchi = xy00_batchi.float().cuda()
        xy10_batchi = xy00_batchi + torch.Tensor([1., 0.]).cuda()
        xy01_batchi = xy00_batchi + torch.Tensor([0., 1.]).cuda()
        xy11_batchi = xy00_batchi + torch.Tensor([1., 1.]).cuda()

        projection = torch.cat([xy00_batchi[:, 1] * w + xy00_batchi[:, 0], xy10_batchi[:, 1] * w + xy10_batchi[:, 0], xy01_batchi[:, 1] * w + xy01_batchi[:, 0], xy11_batchi[:, 1] * w + xy11_batchi[:, 0]], dim=0).long()
        mask1 = torch.zeros(h * w).cuda()
        mask1[projection] = 1
        mask1 = torch.unsqueeze(mask1.view(h, w), 0)
        if i == 0:
          mask0_stack = mask0
          mask1_stack = mask1
        else:
          mask0_stack = torch.cat([mask0_stack, mask0], dim=0)
          mask1_stack = torch.cat([mask1_stack, mask1], dim=0)

      return mask_p, mask0_stack, mask1_stack

def mask_p01_duoci(coords, maskp01_qian):

      b, h, w, _ = coords.size()

      x0 = torch.floor((w - 1) * (coords[:, :, :, 0] / 2.0 + 0.5).float()).float().cuda()
      y0 = torch.floor((h - 1) * (coords[:, :, :, 1] / 2.0 + 0.5).float()).float().cuda()

      maskp01_qian_luoji = torch.eq(maskp01_qian.cuda(), torch.ones_like(maskp01_qian).cuda())
      fuer_like = -2 * torch.ones_like(x0).cuda()

      x0 = torch.where(maskp01_qian_luoji, x0, fuer_like)
      y0 = torch.where(maskp01_qian_luoji, y0, fuer_like)
      xy00 = torch.stack([x0, y0], dim=3).cuda()
      for i in range(b):
        xy00_batchi = xy00[i, :, :, :].view(-1, 2).float().cuda() 
        xy10_batchi = xy00_batchi + torch.Tensor([1., 0.]).cuda()
        xy01_batchi = xy00_batchi + torch.Tensor([0., 1.]).cuda()
        xy11_batchi = xy00_batchi + torch.Tensor([1., 1.]).cuda()

        projection = torch.cat([xy00_batchi[:, 1] * w + xy00_batchi[:, 0], xy10_batchi[:, 1] * w + xy10_batchi[:, 0], xy01_batchi[:, 1] * w + xy01_batchi[:, 0], xy11_batchi[:, 1] * w + xy11_batchi[:, 0]], dim=0).long()
        mask1 = torch.zeros(h * w).cuda()
        mask1[projection] = 1
        mask1 = torch.unsqueeze(mask1.view(h, w), 0)
        if i == 0:
          mask1_stack = mask1
        else:
          mask1_stack = torch.cat([mask1_stack, mask1], dim=0)
      return mask1_stack

        
def pose_vec2mat_revised(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    b, _, _ = transform_mat.size()
    filler = Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])).type_as(transform_mat).expand(b, 1, 4)

    transform_mat = torch.cat([transform_mat, filler], dim=1)# [B, 4, 4]

    return transform_mat

def build_rigid_maskp01(tgt_img, ref_img, depth, depth_ref, current_pose,intrinsics_scaled,intrinsics_scaled_inv, rotation_mode, padding_mode, maskp01_duoci=False):
    
    with torch.no_grad():
        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        _, fwd_mask_p, fwd_mask_0, bwd_mask_1 = inverse_warp_mask(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=True, maskp01_duoci=False, maskp01_qian=None)

        _, bwd_mask_p, bwd_mask_0, fwd_mask_1 = inverse_warp_mask(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=True, maskp01_duoci=False, maskp01_qian=None)
        if maskp01_duoci == True:
            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)

            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)

            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)
        fwd_mask_p = torch.unsqueeze(fwd_mask_p, 1).float()
        bwd_mask_p = torch.unsqueeze(bwd_mask_p, 1).float()
        fwd_mask_0 = torch.unsqueeze(fwd_mask_0, 1).float()
        bwd_mask_0 = torch.unsqueeze(bwd_mask_0, 1).float()
        fwd_mask_1 = torch.unsqueeze(fwd_mask_1, 1).float()
        bwd_mask_1 = torch.unsqueeze(bwd_mask_1, 1).float()

        fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
        bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1
        fwd_mask_p1 = fwd_mask_p *  fwd_mask_1
        bwd_mask_p1 = bwd_mask_p *  bwd_mask_1
        visual_mask =  torch.cat((bwd_mask_p01,fwd_mask_p01,bwd_mask_p1,fwd_mask_p1,bwd_mask_p,fwd_mask_p,bwd_mask_0,fwd_mask_0,bwd_mask_1,fwd_mask_1),1)
    return fwd_mask_p01, bwd_mask_p01

def coor_trans(ori_coor, pose_mat):
    """ 
        ori_coor: [b 3 n]
        pose_mat: [4,4]

        return [b 3 n]
    """ 
    b,_,n = ori_coor.size()
    rot_mat = pose_mat[:,:3,:3]
    trans_mat = pose_mat[:,:3,3:4]

    new_coor = rot_mat.bmm(ori_coor)
    return new_coor+trans_mat   # [b 3 n]
    
def build_angle_sky(pc, velo2cam, cam2velo):
    """ 
        cam: [b 3  h w]
        A2B: [4,4]
    """ 
    b, c, h, w = pc.size()
    cam_flatten = pc.view(b, 3, -1)
    velo_coor = coor_trans(cam_flatten, cam2velo)

    x_coor = velo_coor[:, 0, :]               # [b h*w]
    z_coor = velo_coor[:, 2,:]                    # [b h*w]
    arc_coor = torch.atan(z_coor/x_coor)        # [b h*w]
    mask = arc_coor<0.035
    mask = mask + 0.0              # [b  h*w]
   
    return mask.view(b, h, w)



def build_sky_mask(depth):
    ''' 
        input: 
            depth [b h w]
        output: 
            mask []
    '''
    b, h, w = depth.size() 
    sky_mask = torch.zeros([b, h, w]).cuda()
    for i in range(b):
        temp_depth = depth[i,:,:]   # [h, w]
        max_depth = temp_depth.max()
        valid_depth = temp_depth<35
        sky_mask [i,:,:] = valid_depth

    return sky_mask

def build_groud_mask(pc):
    ''' 
        input: 
            pc [b 3 h w]
        output: 
            mask []
    '''
    b, _, h, w = pc.size() 
    groud_mask = torch.zeros([b, h, w]).cuda()
    for i in range(b):
        temp_pc_y = pc[i,1,:,:]   # [h, w]
        max_y = temp_pc_y.max()
        
        valid_depth = temp_pc_y< 1.15
        groud_mask [i,:,:] = valid_depth

    return groud_mask

def BackprojectDepth(depth, inv_k):
    ''' 
        depth: [b, 1, h, w]
        invk: [b 3 3]
    ''' 
    b, _, h, w = depth.size()
    meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords), requires_grad=False)
    ones = nn.Parameter(torch.ones(b, 1, h * w), requires_grad=False).cuda()
    pix_coords = torch.unsqueeze(torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(b, 1, 1)
    pix_coords = nn.Parameter(torch.cat([pix_coords.cuda(), ones], 1), requires_grad=False)
    cam_points = torch.matmul(inv_k, pix_coords.cuda())
    cam_points = depth.view(b, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)
    cam_points = cam_points[:, :3,:].view(b, 3, h, w)
    return cam_points


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode)

    return projected_img, valid_mask, projected_depth, computed_depth


def build_block_mask(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, inv_intrinsic):
    """ 
        return [b h w]
    """ 
    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic)
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)    # 1 means blocked
    valid_mask = diff_depth<0.5     # [b 1 h w]
    return valid_mask.squeeze(1)

