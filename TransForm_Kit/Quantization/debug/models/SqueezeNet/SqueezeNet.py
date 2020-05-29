#!/usr/bin/env python
# coding: utf-8

import numpy as np
import struct
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from torch.autograd import Variable
from collections import OrderedDict

class Fire(nn.Module):
    
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(squeeze_planes, eps=0.001, momentum=0.1, affine=True)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expand1x1_planes, eps=0.001, momentum=0.1, affine=True)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expand3x3_planes, eps=0.001, momentum=0.1, affine=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.squeeze(x)
        x = self.bn1(x)
        x = self.squeeze_activation(x)
        x1 = self.expand1x1(x)
        x1 = self.bn2(x1)
        x1 = self.expand1x1_activation(x1)
        x2 = self.expand3x3(x)
        x2 = self.bn3(x2)
        x2 = self.expand3x3_activation(x2)
        x3 = torch.cat([x1,x2],1)
        return x3

class SqueezeNet(nn.Module):
    
    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True)
            self.relu1 = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire2 = Fire(64, 16, 64, 64)
            self.fire3 = Fire(128, 16, 64, 64)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire4 = Fire(128, 32, 128, 128)
            self.fire5 = Fire(256, 32, 128, 128)
            self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire6 = Fire(256, 48, 192, 192)
            self.fire7 = Fire(384, 48, 192, 192)
            self.fire8 = Fire(384, 64, 256, 256)
            self.fire9 = Fire(512, 64, 256, 256)
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                            "1_0 or 1_1 expected".format(version=version))
        self.dropout = nn.Dropout(p=0.5)
        self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_classes, 128, bias=False)
        self.bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.maxpool2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.maxpool3(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze(x,3)
        x = self.bn(x)
        x = torch.squeeze(x)
        return x

