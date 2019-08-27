
# coding: utf-8


import math
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
class ResNet50PreProcess(object):
    def __call__(self,image):
        w = 224
        h = 224
        image = np.array(image)
        img_matlab = image.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        imgFloat = img_matlab.astype(float)
        imgResize = cv2.resize(imgFloat,(w,h))
        imgResize[:,:,0] = imgResize[:,:,0] - 110.177
        imgResize[:,:,1] = imgResize[:,:,1] - 117.644
        imgResize[:,:,2] = imgResize[:,:,2] - 117.378
        imgProc = imgResize
        imgProc = np.swapaxes(imgProc, 0, 2)
        imgProc = np.swapaxes(imgProc, 1, 2)
        imgProc = torch.Tensor(imgProc)
        return imgProc

# ResNet50
def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        identity = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) # the python do not only transfer the pars, but also transfer the function!
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(inplanes,planes,stride)#no stride
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)#stride
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity) # attention: the par is not out, it is 'x'!
        out += identity
        out = self.relu(out)        
        return out

class ResNet(nn.Module):
    
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False):# zero?
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)#bias=False
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)#padding = 1
        
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m,BasicBolck):
                    nn.init.constant_(m.bn2.weight,0)
                    
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                nn.BatchNorm2d(planes*block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        pd = (0,1,0,1)
        x = F.pad(x,pd,'constant',0)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
    def resnet50(pretrained=False,**kwargs):
        model = ResNet(Bottleneck,[3,4,6,3],**kwargs)
        return model
