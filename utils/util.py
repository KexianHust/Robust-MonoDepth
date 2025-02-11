#!/usr/bin/env python3
# coding: utf-8

'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2019/04/10
'''

import torch
import h5py
import numpy as np
import math
import os


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
