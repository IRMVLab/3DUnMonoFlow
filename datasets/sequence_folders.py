import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import custom_transforms
import torch


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def crawl_folders(folders_list, sequence_length):
    sequence_set = []
    demi_length = (sequence_length - 1) // 2
    for folder in folders_list:
        intrinsics = np.genfromtxt(folder / 'cam.txt').astype(np.float32).reshape((3, 3))
        calib_file = read_raw_calib_file(folder/'calib.txt')
        velo2cam = calib_file['Tr']
        velo2cam = velo2cam.reshape(3, 4)
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        imgs = sorted(folder.files('*.jpg'))
        gts = sorted(folder.files('*.npy'))
        if len(imgs) < sequence_length:
            continue
        for i in range(demi_length, len(imgs) - demi_length):
            sample = {'intrinsics': intrinsics, 'velo2cam':velo2cam, 'tgt': imgs[i], 'ref_imgs': [], 'ground_truth': gts[i]}
            for j in range(-demi_length, demi_length + 1):
                if j != 0:
                    sample['ref_imgs'].append(imgs[i + j])
            sequence_set.append(sample)
    random.shuffle(sequence_set)
    return sequence_set


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, transform_1=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / 'train.txt' if train else self.root / 'val.txt'
        self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.samples = crawl_folders(self.scenes, sequence_length)
        self.transform = transform
        self.transform_1 = transform_1

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        depth_gt = np.load(sample['ground_truth'])

        velo2cam = np.copy(sample['velo2cam'])

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            imgs_1, intrinsics_1 = self.transform_1([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:3]
            tgt_img_1 = imgs_1[0]
            ref_imgs_1 = imgs_1[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),velo2cam, np.linalg.inv(velo2cam),tgt_img_1, ref_imgs_1, depth_gt

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':

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
        './../temp_kitti_256',
        transform=train_transform,
        transform_1=train_transform_1,
        seed=0,
        train=True,
        sequence_length=3
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True)

    for i, (tgt_img, ref_imgs, intrinsics, inv_intrinsics, ori_tgt_image, ori_ref_imgs, depth_gt) in enumerate(train_loader):
        print(tgt_img.size())
        print(ref_imgs[0].size())
        print(intrinsics.size())
        print(inv_intrinsics.size())
        print(ori_tgt_image.size())
        print(ori_ref_imgs[0].size())
        print(depth_gt.size())
        print(len(ref_imgs))
        print(len(ori_ref_imgs))
        a = np.median(depth_gt)
        sum_depth = depth_gt.sum()
        print("depth sum", sum_depth)
        print(depth_gt)
