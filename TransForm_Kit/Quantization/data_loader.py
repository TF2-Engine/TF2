# Copyright 2019 Inspur Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
from collections import OrderedDict
import torch.utils.data as data
import numpy as np
from data.SSD import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform, MEANS
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

class GoogLeNetPreProcess(object):
    def __call__(self,image):
        w = 256
        h = 256
        cropsize = 224
        image = np.array(image)
        img_matlab = image.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        imgResize = cv2.resize(img_matlab,(w,h))
        imgFloat = imgResize.astype(float)
        imgFloat[:,:,0] = imgFloat[:,:,0] - 104
        imgFloat[:,:,1] = imgFloat[:,:,1] - 117
        imgFloat[:,:,2] = imgFloat[:,:,2] - 123
        w_off = int((w-cropsize)/2)
        h_off = int((h-cropsize)/2)
        imgCrop = imgFloat[h_off:(h_off+cropsize),w_off:(w_off+cropsize)]
        imgProc = imgCrop
        imgProc = np.swapaxes(imgProc, 0, 2)
        imgProc = np.swapaxes(imgProc, 1, 2)
        imgProc = torch.Tensor(imgProc)
        return imgProc

class SqueezeNetPreProcess(object):
    def __call__(self,image):
        image = np.float32(image)
        img_matlab = image.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        imgProc = (img_matlab-127.5)/128.0
        imgProc = np.swapaxes(imgProc, 0, 2)
        imgProc = np.swapaxes(imgProc, 1, 2)
        image = torch.Tensor(image)
        return imgProc

def load_data(net_name):
    if net_name == 'resnet50':
        data_path = '/data/yutong/imagenet'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]) 
        val = datasets.ImageFolder(os.path.join(data_path,'val'),transform)
        ValLoader = data.DataLoader(val,batch_size=1,shuffle=True)
        return ValLoader
    if net_name == 'googlenet':
        data_path = 'data'
        transform = transforms.Compose([GoogLeNetPreProcess()])
        val = datasets.ImageFolder(os.path.join(data_path,'val'),transform)
        ValLoader = data.DataLoader(val,batch_size=1,shuffle=True)
        return ValLoader
    if net_name == 'squeezenet':
        data_path = 'data'
        transform = transforms.Compose([SqueezeNetPreProcess()])
        val = datasets.ImageFolder(os.path.join(data_path,'test_images_aligned_test'),transform)
        ValLoader = data.DataLoader(val,batch_size=1,shuffle=True)
        return ValLoader
    if net_name == 'ssd':
        ValLoader = []
        ssd_dataset = VOCDetection(VOC_ROOT, [('2007', 'test')], BaseTransform(300, MEANS), VOCAnnotationTransform())
        for i in range(1000):
            test_x, test_y = ssd_dataset.pull_item(i) 
            test_x = Variable(test_x.unsqueeze(0))
            ValLoader.append((test_x,test_y))
        return ValLoader


