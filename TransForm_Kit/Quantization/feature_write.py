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
import numpy as np
import struct
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
import os
from net_structure import structure_hook
from data_loader import load_data
from model_loader import load_model
import sys
def FeatureWrite(name,x):
    global feature_path
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    dim = len(x.shape)
    if dim == 1:
        with open(feature_path + '/' +name + '.bin','wb') as data:
            for i in range(x.shape[0]):
                    st = x[i].item()
                    st = struct.pack('f',st)
                    data.write(st)
    if dim == 2:
        with open(feature_path + '/' +name + '.bin','wb') as data:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    st = x[i][j].item()
                    st = struct.pack('f',st)
                    data.write(st)
    if dim == 3:
        with open(feature_path + '/' +name + '.bin','wb') as data:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        st = x[i][j][k].item()
                        st = struct.pack('f',st)
                        data.write(st)
    if dim == 4:
        with open(feature_path + '/' +name + '.bin','wb') as data:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        for l in range(x.shape[3]):
                            st = x[i][j][k][l].item()
                            st = struct.pack('f',st)
                            data.write(st)

def feature_hook(model, x):
    def register_hook(module):
        def hook(module, input, output):
            global Features, layer_name, layer_count, call_count, threshold
            Features[layer_count] = np.maximum(abs(output.detach().numpy()),Features[layer_count])
            if call_count == threshold:
                FeatureWrite(layer_name[layer_count],Features[layer_count])
            layer_count = layer_count + 1
        hooks.append(module.register_forward_hook(hook))
    hooks = []
    model.apply(register_hook)
    model(x)
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    
    layer_count = 0
    call_count = 0
    threshold = 999
    # set your network will be quantized: googlenet', 'resnet50', 'squeezenet', 'ssd' ...
    net_name = sys.argv[1]
    ValLoader = load_data(net_name)
    cnn, layer_name, Features, feature_path = load_model(net_name)
    for j,(test_x,test_y) in enumerate(ValLoader):
        feature_hook(cnn,test_x)
        layer_count = 0
        call_count = call_count + 1
        print(j)
        if j%threshold == 0 and j != 0:
            print('The 1000 pic. features have been written in '+feature_path+'.')
            break
 
 
