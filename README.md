# 3DUnMonoFlow


Code for the paper:

*G. Wang, X. Tian, R. Ding, and H. Wang,*  Unsupervised Learning of Scene Flow from Monocular Camera, in International Conference on Robotics and Automation, 2021.

## Prerequisites

Our model is trained and tested under:

- Python 3.6.9
- NVIDIA GPU + CUDA CuDNN
- PyTorch (torch == 1.7.1)
- torchvision
- scipy
- argparse
- tensorboardX
- tqdm
- numba
- cffi

Operation in this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch) is used to compile the furthest point sampling, grouping and gathering operation for PyTorch.

```bash
cd pointnet2
python setup.py install
cd ../
```

## Data preprocess

We referred to the dataset pre-processing approach in this [repo](https://github.com/JiawangBian/SC-SfMLearner-Release) to generate our data for training from KITTI Odometry dataset. We uploaded our processed dataset and you can click this [link]() to download.

KITTI scene flow dataset is used for test and evaluation. [Flownet3d](https://github.com/xingyul/flownet3d)  processed the first 150 data points from KITTI scene flow dataset and removed the ground points. We copied the link [here](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) for you to download.

## Get started

### Train

Set `data` in the configuration file to the path where your training data is saved and change the `batch_size` according to your GPU. Then run:

```bash
python train.py config.yaml
```

Our pre-trained models are provided [here](https://drive.google.com/drive/folders/1fA1LHxJhLHzkqAmCOtfe5mMkDHRN5RQS?usp=sharing) for download.

### Evaluate

Set `evaluate` in the configuration file to be `True`. Then run:

```bash
python train.py config.yaml
```

## Acknowledgement

We are grateful for [repo](https://github.com/DylanWusee/PointPWC) for its github repository.