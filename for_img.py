import sys
import cmd_args
import numpy as np
from datasets.sequence_folders import SequenceFolder
from datasets.kitti_data import SceneflowDataset
import cv2
from tensorboardX import SummaryWriter
import os
import csv
import custom_transforms
import torch
import models
import torch.backends.cudnn as cudnn
from itertools import chain
from utils.inverse_warp import pixel2cam, build_sky_mask, BackprojectDepth, build_block_mask, build_groud_mask, build_angle_sky
from utils.external_util import devide_by_index, set_by_index, AverageMeter, readlines, tensor2array, cam2cam, write_ply, pose_vec2mat, scale_con
from torch.autograd import Variable
import scipy.misc
import transforms
import random
from models.HALFlow import multiScaleChamferSmoothCurvature,rgbConsistencyLoss

velo2cam = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03],
                     [-2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02],
                     [9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01],
                     [0, 0, 0, 1]])
n_iter = 0
import matplotlib.pyplot as plt

def _gray2rgb(im, cmap='plasma'):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img

def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap='plasma'):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp

def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f * -1.0 + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y


def get_2d_flow(pc1, pc2, predicted_pc2, paths=None):
    if paths == None:
        px1, py1 = project_3d_to_2d(pc1)
        px2, py2 = project_3d_to_2d(predicted_pc2)
        px2_gt, py2_gt = project_3d_to_2d(pc2)
        flow_x = (px2 - px1).cpu().detach().numpy()
        flow_y = (py2 - py1).cpu().detach().numpy()

        flow_x_gt = (px2_gt - px1).cpu().detach().numpy()
        flow_y_gt = (py2_gt - py1).cpu().detach().numpy()

    else:
        focallengths = []
        cxs = []
        cys = []
        constx = []
        consty = []
        constz = []
        for path in paths:
            fname = os.path.split(path)[-1]
            calib_path = os.path.join(
                os.path.dirname(__file__),
                'utils',
                'calib_cam_to_cam',
                fname + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)
                focallengths.append(-P_rect_left[0, 0])
                cxs.append(P_rect_left[0, 2])
                cys.append(P_rect_left[1, 2])
                constx.append(P_rect_left[0, 3])
                consty.append(P_rect_left[1, 3])
                constz.append(P_rect_left[2, 3])
        focallengths = np.array(focallengths)[:, None, None]
        cxs = np.array(cxs)[:, None, None]
        cys = np.array(cys)[:, None, None]
        constx = np.array(constx)[:, None, None]
        consty = np.array(consty)[:, None, None]
        constz = np.array(constz)[:, None, None]

        px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                          constx=constx, consty=consty, constz=constz)

        flow_x = (px2 - px1)
        flow_y = (py2 - py1)

        flow_x_gt = (px2_gt - px1)
        flow_y_gt = (py2_gt - py1)

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)

    return flow_pred, flow_gt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


def testOnce(scene_net, test_loader):
    global args
    scene_net.eval()
    epe_ori = AverageMeter()
    acc1_ori = AverageMeter()
    acc2_ori = AverageMeter()
    outliers = AverageMeter()

    epe2dAG = AverageMeter()
    acc2dAG = AverageMeter()

    for i, (pos1, pos2, color1, color2, flow, mask) in enumerate(test_loader):
        pos1 = Variable(pos1.cuda()).permute(0, 2, 1).float()
        # print('pose1:', pos1)
        pos2 = Variable(pos2.cuda()).permute(0, 2, 1).float()
        color1 = Variable(color1.cuda()).permute(0, 2, 1).float()
        color2 = Variable(color2.cuda()).permute(0, 2, 1).float()
        b, c, n = pos1.size()
        flow = Variable(flow.cuda()).permute(0, 2, 1).float()

        trans = torch.tensor([[0, -1, 0],
                              [0, 0, -1],
                              [1, 0, 0]])
        trans_mat = trans.unsqueeze(0).repeat(b, 1, 1).type_as(pos1)
        pos1 = trans_mat.bmm(pos1)
        pos2 = trans_mat.bmm(pos2)
        # color1 = trans_mat.bmm(color1)
        # color2 = trans_mat.bmm(color2)
        flow = trans_mat.bmm(flow).permute(0, 2, 1)

        pred_sfs, _, _ = scene_net(pos1, pos2, color1, color2)  # [b 2048 3]
        pred_sf = pred_sfs[0].permute(0, 2, 1)
        pre = pred_sf.cpu().detach().numpy()
        tar = flow.cpu().detach().numpy()[:, :args.num_points, :]

        # ori baseline -------------------
        l2_norm = np.linalg.norm(np.abs(tar - pre) + 1e-20, axis=-1)
        EPE3D_ori = l2_norm.mean()

        sf_norm = np.linalg.norm(tar, axis=-1)
        relative_err = l2_norm / (sf_norm + 1e-4)

        acc3d_strict_ori = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
        acc3d_relax_ori = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()
        epe_ori.update(EPE3D_ori, args.batch_size)
        acc1_ori.update(acc3d_strict_ori, args.batch_size)
        acc2_ori.update(acc3d_relax_ori, args.batch_size)
        outliers.update(outlier, args.batch_size)

        # 2d acc
        flow_pred_2d, flow_gt_2d = get_2d_flow(pos1[:, :, :args.num_points], pos1[:, :, :args.num_points] + flow[:, :args.num_points, :].permute(0, 2, 1),
                                               pos1[:, :, :args.num_points] + pred_sf[:, :args.num_points, :].permute(0, 2, 1))
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)
        # print("epe2d: ", EPE2D)
        # print("acc2d: ", acc2d)
        epe2dAG.update(EPE2D, args.batch_size)
        acc2dAG.update(acc2d, args.batch_size)

    return epe_ori.avg[0], acc1_ori.avg[0], acc2_ori.avg[0], epe2dAG.avg[0], acc2dAG.avg[0], outliers.avg[0]


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1 / disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def robust_l1_per_pix(x, q=0.5, eps=1e-2, compute_type=False, q2=0.5, eps2=1e-2):
    if compute_type:
        x = torch.pow((x.pow(2) + eps), q)
    else:
        x = torch.pow((x.abs() + eps2), q2)
    return x


def main():
    global args, n_iter
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    print(args)
    best_loss = float('inf')
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    log_dir = args.save_path + '/' + args.name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('=> will save everything to {}'.format(log_dir))

    training_writer = SummaryWriter(log_dir)
    with open(log_dir + '/log.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'eva_epe', 'eva_acc1', 'eva_acc2', 'epe_2d', 'acc_2d', 'outlier'])

    torch.manual_seed(args.seed)
    os.system('cp %s %s' % ('train.py', log_dir))
    os.system('cp %s %s' % ('config.yaml', log_dir))
    depth_weight_path = os.path.join(args.load_weights_folder, "dispnet_model_best.pth.tar")
    pose_weight_path = os.path.join(args.load_weights_folder, "exp_pose_model_best.pth.tar")

    # load train data -------------------------
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    train_transform_1 = custom_transforms.Compose([
        custom_transforms.ArrayToTensor()
    ])

    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        transform_1=train_transform_1,
        seed=None,   #args.seed,
        train=True,
        sequence_length=args.sequence_length
    )
    print("train data len set: ", len(train_set))
    print('{} samples found in train data'.format(len(train_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # load test data  -----------------
    test_set = SceneflowDataset(args.test_data_path, npoints=args.index_num, train=False)
    print("len set: ", len(test_set))
    print('{} samples found in test data'.format(len(test_set)))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=True
    )
    # create model -------------------
    scene_net = getattr(models, args.flownet_name)(args.flag_big_radius, args.kernel_shape, args.layers, args.is_training).cuda()

    disp_net = getattr(models, args.depth_name)(args.resnet_layers, 0).cuda()
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)
    disp_net.eval()

    pose_net = getattr(models, args.pose_name)(18, False).cuda()
    weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()

    if args.resume:
        print("=> resuming from checkpoint %s " % args.resume_name)
        resume_path = './pretrained_weight/' + args.resume_name
        print("load model from path: ", resume_path)
        scenenet_weights = torch.load(resume_path)
        scene_net.load_state_dict(scenenet_weights['state_dict'])
        print("load resume model successfully! ----------------")

    cudnn.benchmark = True

    if args.multi_gpu:
        scene_net = torch.nn.DataParallel(scene_net)

    parameters = chain(scene_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay
                                 )

    if args.evaluate:
        epe, acc1, acc2, epe2d, acc2d, outlier = testOnce(scene_net, test_loader)
        print("epe without mask: ", epe)
        print("acc strict without mask: ", acc1)
        print("acc relax without mask: ", acc2)
        print("epe2d with mask: ", epe2d)
        print("acc2d strict with mask: ", acc2d)


        with open(args.save_path + '/' + args.name + '/log.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([0, epe, acc1, acc2, epe2d, acc2d, outlier])
    else:

        for epoch in range(args.epochs):
            print("current epoch: ", epoch)
            train_loss = train(train_loader, disp_net, pose_net, scene_net, optimizer, training_writer)
            training_writer.add_scalar("epoch train loss", train_loss, epoch)
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({'epoch': epoch + 1, 'state_dict': scene_net.state_dict()}, args.save_path + args.name + '/model.best.t7')
            torch.save({'epoch': epoch + 1, 'state_dict': scene_net.state_dict()}, args.save_path + args.name + '/scene_model.newest.t7')



def train(train_loader, disp_net, pose_net, scene_net, optimizer, training_writer=None):
    global n_iter, args
    disp_net.eval()
    pose_net.eval()
    scene_net.train()

    losses = AverageMeter(precision=4)
    global velo2cam
    cam2velo = np.linalg.inv(velo2cam)
    velo2cam_ten = torch.from_numpy(velo2cam).to(device)
    cam2velo_ten = torch.from_numpy(cam2velo).to(device)

    for i, (tgt_img, ref_imgs, intrinsics, inv_intrinsics, ori_tgt_img, ori_ref_imgs, depth_gt) in enumerate(train_loader):

        tgt_img = tgt_img.to(device)  # [b 3 h w]
        ref_imgs = [img.to(device) for img in ref_imgs]  # list 2 [b 3 h w]
        intrinsic = intrinsics.to(device)  # [b 3 3]
        inv_intrinsic = inv_intrinsics.to(device)  # [b 3 3]

        ori_tgt_img = ori_tgt_img.to(device)  # [b 3 h w]
        ori_ref_imgs = [img.to(device) for img in ori_ref_imgs]  # list 2 [b 3 h w]
        b, _, h, w = tgt_img.size()

        depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        depth = depth[0]
        # print('depth size', depth.size())

        fw_depth = ref_depths[1][0]
        bw_depth = ref_depths[0][0]

        pose_predict = pose_net(tgt_img, ref_imgs[1])
        pose_mat = pose_vec2mat(pose_predict)
        pose_inv_predict = pose_net(ref_imgs[1], tgt_img)
        # print('pose size:', pose_predict.size())

        # remove_sky
        new_pred_depth, new_fw_depth, pose_mat = scale_con(depth_gt, depth.squeeze(1), fw_depth.squeeze(1), pose_mat)
        sky_mask1 = build_sky_mask(new_pred_depth.squeeze(1))  # [b h w]
        sky_mask2 = build_sky_mask(new_fw_depth.squeeze(1))
        pc1_cam1 = BackprojectDepth(new_pred_depth, inv_intrinsic)
        pc2_cam2 = BackprojectDepth(new_fw_depth, inv_intrinsic)
        ground_mask1 = build_groud_mask(pc1_cam1)
        ground_mask2 = build_groud_mask(pc2_cam2)
        angle_sky_mask = build_angle_sky(pc1_cam1, velo2cam_ten.type_as(pc1_cam1), cam2velo_ten.type_as(pc1_cam1))
        block_mask1 = build_block_mask(tgt_img, ref_imgs[1], depth, fw_depth, pose_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)
        block_mask2 = build_block_mask(ref_imgs[1], tgt_img, fw_depth, depth, pose_inv_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)

        index1 = np.zeros((b, args.index_num))
        index2 = np.zeros((b, args.index_num))
        # sample 8192 points randomly for the beginning epochs
        training_writer.add_image('sky_mask', tensor2array(sky_mask1.data[0].cpu(), max_value=1, colormap='bone'), n_iter)
        training_writer.add_image('block_mask_1', tensor2array(block_mask1.data[0].cpu(), max_value=1, colormap='bone'), n_iter)
        training_writer.add_image('block_mask_2', tensor2array(block_mask2.data[0].cpu(), max_value=1, colormap='bone'), n_iter)

        total_mask = sky_mask1 * sky_mask2 * block_mask1 * block_mask2 * ground_mask1 * ground_mask2 * angle_sky_mask.type_as(ground_mask2)
        training_writer.add_image('total_mask', tensor2array(total_mask.data[0].cpu(), max_value=1, colormap='bone'), n_iter)

        total_mask_sum = total_mask.sum(dim=[1, 2])
        total_mask_sum = (total_mask_sum < args.index_num)
        total_mask_flatten = total_mask.view(b, -1)
        if total_mask_sum.sum().item() > 0:
            continue
        else:
            for i in range(b):
                f = (total_mask_flatten[i, :] > 0).nonzero()
                f = f.view(-1)
                n_f = f.cpu().numpy()
                np.random.shuffle(n_f)
                index1[i, :] = n_f[:args.index_num]
                np.random.shuffle(n_f)
                index2[i, :] = n_f[:args.index_num]


                # temp_pc1_cam1 = pixel2cam(new_pred_depth.squeeze(1), inv_intrinsic)
        pc1_cam1_flatten = pc1_cam1.view(b, 3, -1)
        # temp_pc1_cam1_flatten = temp_pc1_cam1.view(b, 3, -1)
        rgb1_flatten = tgt_img.view(b, 3, -1)
        ori_rgb1_flatten = ori_tgt_img.view(b, 3, -1)

        pc1_cam1_part = devide_by_index(pc1_cam1_flatten, index1).contiguous()
        rgb1_part = devide_by_index(rgb1_flatten, index1).contiguous()
        # use_pc1_cam1 = pc1_cam1_part[:, :,:args.num_points]
        ori_rgb1_part = devide_by_index(ori_rgb1_flatten, index1).contiguous()
        # use_rgb1 = ori_rgb1_part[:, :, :args.num_points]

        # temp_pc2_cam2 = pixel2cam(new_fw_depth.squeeze(1), inv_intrinsic)
        pc2_cam2_flatten = pc2_cam2.view(b, 3, -1)
        # temp_pc2_cam2_flatten = temp_pc2_cam2.view(b, 3, -1)
        rgb2_flatten = ref_imgs[1].view(b, 3, -1)
        ori_rgb2_flatten = ori_ref_imgs[1].view(b, 3, -1)

        pc2_cam2_part = devide_by_index(pc2_cam2_flatten, index2).contiguous()

        rgb2_part = devide_by_index(rgb2_flatten, index2).contiguous()
        # use_pc2_cam2 = pc2_cam2_part[:,:,:args.num_points]
        ori_rgb2_part = devide_by_index(ori_rgb2_flatten, index2).contiguous()
        # use_rbg2 = ori_rgb2_part[:, :, :args.num_points]

        fea1 = rgb1_part
        fea2 = rgb2_part
        ori_fea1 = ori_rgb1_part
        ori_fea2 = ori_rgb2_part
        over_all_flows, idx1s, idx2s = scene_net(pc1_cam1_part, pc2_cam2_part, fea1, fea2)  # [b 3 2048]
        idx1 = idx1s[0]
        idx2 = idx2s[0]
        pc1_2048 = pc1_cam1_part[:, :, idx1[0]]
        pc2_2048 = pc2_cam2_part[:, :, idx2[0]]
        ori_rgb1_2048 = ori_rgb1_part[:, :, idx1[0]]
        ori_rgb2_2048 = ori_rgb2_part[:, :, idx2[0]]
        sf_pc2 =  pc1_2048 + over_all_flows[0]

        depth1 = np.transpose(tensor2array(depth[0].data.cpu(), max_value=10), (1,2,0))[:,:,1]
        depth2 = np.transpose(tensor2array(fw_depth[0].data.cpu(), max_value=10), (1,2,0))[:,:,1]
        depth1 = _normalize_depth_for_display(depth1, cmap='plasma')
        depth2 = _normalize_depth_for_display(depth2, cmap='plasma')

        # write ply
        if n_iter <1000:
            cv2.imwrite('./../visualization/tgt_img/'+ str(n_iter)+'.png', 255*ori_tgt_img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()[:,:,[2,1,0]])
            cv2.imwrite('./../visualization/fw_img/'+ str(n_iter)+'.png', 255*ori_ref_imgs[1][0,:,:,:].permute(1,2,0).cpu().detach().numpy()[:,:,[2,1,0]])
            de = tensor2array(depth[0].data.cpu(), max_value=10)

            scipy.misc.imsave('./../visualization/tgt_depth/'+ str(n_iter)+'.png', depth1)
            scipy.misc.imsave('./../visualization/fw_depth/'+ str(n_iter)+'.png', depth2)
            
            pc1_name = './../visualization/pc1/'+str(n_iter)+'.ply'
            pc2_name = './../visualization/pc2/'+str(n_iter)+'.ply'
            pc1_part_name = './../visualization/part_pc1/'+str(n_iter)+'.ply'
            pc2_part_name = './../visualization/part_pc2/'+str(n_iter)+'.ply'
            pc1_2048_name = './../visualization/2048_pc1/' + str(n_iter) + '.ply'
            pc2_2048_name = './../visualization/2048_pc2/' + str(n_iter) + '.ply'
            sf_pc2_name = './../visualization/sf_pc2/' + str(n_iter) + '.ply'

            write_ply(pc1_name, pc1_cam1_flatten[0].permute(1,0), 255*ori_rgb1_flatten[0].permute(1,0))
            write_ply(pc2_name, pc2_cam2_flatten[0].permute(1,0), 255*ori_rgb2_flatten[0].permute(1,0))
            write_ply(pc1_part_name, pc1_cam1_part[0].permute(1,0), 255*ori_rgb1_part[0].permute(1,0))
            write_ply(pc2_part_name, pc2_cam2_part[0].permute(1,0),  255*ori_rgb2_part[0].permute(1,0))
            write_ply(pc1_2048_name, pc1_2048[0].permute(1, 0), 255 * ori_rgb1_2048[0].permute(1, 0))
            write_ply(pc2_2048_name, pc2_2048[0].permute(1, 0), 255 * ori_rgb2_2048[0].permute(1, 0))
            write_ply(sf_pc2_name, sf_pc2[0].permute(1, 0), 255 * ori_rgb1_2048[0].permute(1, 0))
            print('finish :' + str(n_iter))
        n_iter+=1
        

    return 0


if __name__ == '__main__':
    print("run train.py, start here: ---------")
    main()