#!/usr/bin/env python3
# coding: utf-8

'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2020/07/24
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import resnet, resnext, resnest, resnext101_wsl, convnext

from networks import *

class Decoder(nn.Module):
    def __init__(self, inchannels = [256, 512, 1024, 2048], midchannels = [256, 256, 256, 512], upfactors = [2,2,2,2], outchannels = 1):
        super(Decoder, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.upfactors = upfactors
        self.outchannels = outchannels

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)

        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])

        self.outconv = AO(inchannels=self.inchannels[0], outchannels=self.outchannels, upfactor=2)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                #init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                #init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, features):
        _,_,h,w = features[3].size()
        x = self.conv(features[3])
        x = self.conv1(x)
        x = self.upsample(x)

        x = self.ffm2(features[2], x)
        #print('ffm2:', x.size())
        x = self.ffm1(features[1], x)
        #print('ffm1:', x.size())
        x = self.ffm0(features[0], x)
        #print('ffm0:', x.size())
        #-----------------------------------------
        x = self.outconv(x)

        return x

class DepthNet(nn.Module):
    __resnet_factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152
    }
    __resnext_factory = {
        50: resnext.resnext50,
        101: resnext.resnext101,
    }
    __resnest_factory = {
        50: resnest.resnest50,
        101: resnest.resnest101,
        200: resnest.resnest200,
    }
    def __init__(self,
                backbone='resnet',
                depth=50,
                pretrained=True,
                inchannels=[256, 512, 1024, 2048],
                midchannels=[256, 256, 256, 256], #[256, 256, 256, 512]
                upfactors=[2, 2, 2, 2],
                outchannels=1):
        super(DepthNet, self).__init__()
        self.backbone = backbone
        self.depth = depth
        self.pretrained = pretrained
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.upfactors = upfactors
        self.outchannels = outchannels

        # Build model
        if self.backbone == 'resnet':
            if self.depth not in DepthNet.__resnet_factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__resnet_factory[depth](pretrained=pretrained)
        elif self.backbone == 'resnext':
            if self.depth not in DepthNet.__resnext_factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__resnext_factory[depth](pretrained=pretrained)
        elif self.backbone == 'resnest':
            if self.depth not in DepthNet.__resnest_factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__resnest_factory[depth](pretrained=pretrained)
        elif self.backbone == 'resnext101_wsl':
            self.encoder = resnext101_wsl.resnext101_wsl()
        elif self.backbone == 'convnext_tiny':
            self.encoder = convnext.convnext_tiny(pretrained=pretrained)
        elif self.backbone == 'convnext_small':
            self.encoder = convnext.convnext_small(pretrained=pretrained)
        elif self.backbone == 'convnext_base':
            self.encoder = convnext.convnext_base(pretrained=pretrained)
        elif self.backbone == 'convnext_large':
            self.encoder = convnext.convnext_large(pretrained=pretrained)
        elif self.backbone == 'convnext_xlarge':
            self.encoder = convnext.convnext_xlarge(pretrained=pretrained)

        self.decoder = Decoder(inchannels=self.inchannels, midchannels=self.midchannels, upfactors=self.upfactors, outchannels=self.outchannels)

    def forward(self, x):
        x = self.encoder(x) # 1/4, 1/8, 1/16, 1/32
        x = self.decoder(x)

        return x

if __name__ == '__main__':
    # net = DepthNet(backbone='resnext101_wsl', depth=101, pretrained=True)
    net = DepthNet(backbone='convnext_small', pretrained=True, inchannels=[96, 192, 384, 768], midchannels=[96, 96, 96, 96])
    print(net)
    inputs = torch.ones(4,3,384,384)
    out = net(inputs)
    print(out.size())
