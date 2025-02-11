# -*- coding: utf-8 -*-
"""
Created on Sat April 12 2020
@author: ke xian
"""

import os
import subprocess

import scipy
import numpy as np
import math
import re
import functools
import cv2
import h5py
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops
from sklearn import linear_model
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.distributed as dist
import random

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25

def save_net(file_name, net):
    with h5py.File(file_name, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(file_name, net):
    with h5py.File(file_name, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asaaray(h5f[k]))
            v.copy_(param)

def save_checkpoint(state, root_dir, file_name='checkpoint.pth.tar'):
    torch.save(state, os.path.join(root_dir, file_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(epoch, max_epoch, lr, power=0.9):
    return lr*math.pow(1 - 0.1*epoch/max_epoch, power)


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):
    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret


def plot_learning_curves(net, dir_to_save, types = 'epoch_loss'):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(net.train_loss[types], label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(net.val_loss[types], label='val loss', color='tab:orange')
    ax2.legend(loc = 'upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, types + 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()

def img_reader(path):
    """
    image = Image.open(path)
    if image.mode == 'L':
        image = image.convert('RGB')

    output: range [0,1]
    """
    image = cv2.imread(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    return image.astype(np.float32)

class Img_transformer(object):
    """Transform image to given input
    """
    def __init__(self,
                 width=448,
                 height=448,
                 img_mean=[0.485, 0.456, 0.406],
                 img_std=[0.229, 0.224, 0.225],
                 resize_gt=True,
                 ensure_multiple_of=32,
                 resize_method='upper_bound',
                 interpolation_method=cv2.INTER_CUBIC):
        self.__width = width
        self.__height = height
        self.__img_mean = img_mean
        self.__img_std = img_std
        self.__resize_gt = resize_gt
        self.__ensure_multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__interpolation_method = interpolation_method

    def get_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__ensure_multiple_of) * self.__ensure_multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__ensure_multiple_of) * self.__ensure_multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__ensure_multiple_of) * self.__ensure_multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # get new width and height
        scale_width = self.__width / width
        scale_height = self.__height / height

        if self.__resize_method == 'lower_bound':
            if scale_width > scale_height:
                #fit width
                scale_height = scale_width
            else:
                #fit height
                scale_width = scale_height
            new_height = self.get_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.get_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == 'upper_bound':
            if scale_width < scale_height:
                #fit width
                scale_height = scale_width
            else:
                #fit height
                scale_width = scale_height
            new_height = self.get_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.get_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == 'minimal':
            # scale as least as possible
            if abs(1 - scale_width) < abs(1 - scale_height):
                # fit width
                scale_height = scale_width
            else:
                # fit height
                scale_width = scale_height
            new_height = self.get_multiple_of(scale_height * height)
            new_width = self.get_multiple_of(scale_width * width)
        else:
            raise ValueError('this resize method not implemented')

        return (new_width, new_height)

    def __call__(self, sample, ori_width, ori_height):
        width, height = self.get_size(ori_width, ori_height)

        #resize image
        sample["img"] = cv2.resize(sample["img"], (width, height), interpolation=self.__interpolation_method)
        sample["img"] = (sample["img"] - self.__img_mean) / self.__img_std
        sample["img"] = torch.from_numpy(np.transpose(sample["img"], (2,0,1))).float()

        #resize gt
        if self.__resize_gt:
            sample["gt"] = cv2.resize(sample["gt"], (width, height), interpolation=cv2.INTER_NEAREST)
            sample["gt"] = sample["gt"].astype(np.float32)
            sample["gt"] = np.ascontiguousarray(sample["gt"])
            sample["gt"] = torch.from_numpy(sample["gt"])

        return sample

def norm_disp_map(disp, min_disp, max_disp):
    norm_disp = (disp - min_disp) / (max_disp - min_disp) * 255
    disp_map = norm_disp.astype(np.uint8)

    return disp_map

##########################################
# TUM RGBD
##########################################
def read_tum_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hdf5_file_read:
        img = hdf5_file_read.get('gt/img_1')
        img = np.float32(np.array(img))

        depth_gt = hdf5_file_read.get('gt/gt_depth')
        depth_gt = np.float32(np.array(depth_gt))

        human_mask = hdf5_file_read.get('/gt/human_mask')
        env_mask = 1.0 - np.float32(np.array(human_mask))

    return img, depth_gt, env_mask

##################################
# SINTEL
##################################
def read_sintel_depth(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth

#################################
# distributed training
#################################
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        # args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        # args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        # args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        # os.environ['LOCAL_RANK'] = str(args.gpu)
        # os.environ['RANK'] = str(args.rank)
        # os.environ['WORLD_SIZE'] = str(args.world_size)
        # # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        #node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        print("num_gpus: {}".format(num_gpus))
        args.gpu = proc_id % num_gpus
        #addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = "12263"
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(proc_id)

        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])


    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


################################### LR################################
