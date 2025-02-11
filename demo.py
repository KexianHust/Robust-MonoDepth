#!/usr/bin/env python3
# coding: utf-8

# # Evaluation

'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Create_Date: 2020/09/07
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

import os, argparse, time, sys, logging
import csv, h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import cv2

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append('models')
import DepthNet
from models.transformers.swin_depth import SWINDepthModel
from models.transformers.dpt_depth import DPTDepthModel
from models.transformers.midas_net_custom import MidasNet_small
from utils import *

def main(args):
    # instantiate model
    #net = DepthNet.DepthNet(backbone=args.backbone, depth=args.model_depth)
    if args.backbone == 'dpt_large':
        net = DPTDepthModel(
            backbone="vitl16_384",
            non_negative=True,
        )
    elif args.backbone == 'dpt_hybrid':
        net = DPTDepthModel(
            backbone="vitb_rn50_384",
            non_negative=True,
        )
    elif args.backbone == 'mobileViT':
        net = DPTDepthModel(
            backbone='mobileViT_384',
            non_negative=True,
        )
    elif args.backbone == 'midas_v21_small':
        net = MidasNet_small(features=64, backbone="efficientnet_lite3", exportable=True,
                             non_negative=True, blocks={'expand': True})
    elif args.backbone == 'swin_base':
        net = SWINDepthModel(
            backbone='base12'
        )
    elif args.backbone == 'swinv2_base':
        net = SWINDepthModel(
            backbone='base24'
        )
    else:
        net = DepthNet.DepthNet(backbone=args.backbone, depth=args.model_depth)

    net = nn.DataParallel(net)
    net.eval()
    if args.use_gpu:
        net.cuda()

    # load weights
    restore_dir = args.weights
    if restore_dir is not None:
        checkpoint = torch.load(restore_dir, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(restore_dir))
        print("run without pretrained model")

    sample = {}
    int_width = args.img_size[0]
    int_height = args.img_size[1]

    resize_method = 'minimal' #'upper_bound'

    transformer = Img_transformer(width=int_width,
                                  height=int_height,
                                  img_mean=args.image_mean,
                                  img_std=args.image_std,
                                  resize_gt=False,
                                  ensure_multiple_of=32,
                                  resize_method=resize_method,
                                  interpolation_method=cv2.INTER_CUBIC)

    images = os.listdir('test_imgs')
    num_images = len(images)

    for i, item in enumerate(images):
        img_dir = os.path.join('test_imgs', item)
        print('{}/{}: {}'.format(i + 1, num_images, img_dir))

        img = img_reader(img_dir)
        ori_width, ori_height = img.shape[1], img.shape[0]
        sample["img"] = img
        transformed_sample = transformer(sample, ori_width, ori_height)
        input_img = torch.autograd.Variable(transformed_sample["img"].cuda().unsqueeze(0))

        # forward
        with torch.no_grad():
            output = net(input_img)
        prediction = output.cpu().data
        prediction = F.interpolate(input=prediction, size=(ori_height, ori_width), mode='bicubic', align_corners=False)
        prediction = prediction.squeeze(1)

        # save visual results
        norm_output = norm_disp_map(prediction.numpy(), prediction.min().numpy(), prediction.max().numpy())
        savename = img_dir.split('\\')[-1]
        savename, _ = os.path.splitext(savename)
        plt.imsave(os.path.join(args.result_dir, savename + '.png'), norm_output.squeeze(), cmap='inferno')
        # cv2.imwrite(os.path.join(args.result_dir, savename + '.png'), norm_output.squeeze())



    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SMART Monocular Depth Estimation in PyTorch"
    )
    parser.add_argument(
        "--backbone",
        default="resnet",
        type=str,
        help='backbone: resnet, resnext',
    )
    parser.add_argument(
        "--sampler",
        default="Random",
        type=str,
        help='sampler: Random, Uniform',
    )
    parser.add_argument(
        "--model_depth",
        default=50,
        type=int,
        help='backbone: 50, 101, 152',
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="FILE",
        help="path to weights file",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default='NYUD',
        help="which dataset to test",
        type=str,
    )
    parser.add_argument(
        "--data_aug_ops",
        default='SDA1',
        help="which data augmentation ops",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    args.use_gpu = False
    # seeding for reproducbility
    if torch.cuda.is_available():
        args.use_gpu = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args.epoch = os.path.splitext(os.path.split(args.weights)[-1])[0]
    print(args)
    args.result_dir = 'demo_results/{}_{}/{}'.format(args.backbone, args.data_aug_ops, args.sampler)
    os.makedirs(args.result_dir, exist_ok=True)

    args.img_size = [384, 384]  #[448 448]
    args.image_mean = 0.5 #(0.485, 0.456, 0.406)
    args.image_std = 0.5 #(0.229, 0.224, 0.225)

    main(args)
