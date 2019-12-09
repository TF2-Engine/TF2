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

import torch
import numpy as np
import struct
import math
from net_structure import structure_hook
from model_loader import load_model
import sys, os

def QuantizeLinear(x):
    Max = torch.max(abs(x))
    Limiter = pow(2,n-1) - 1
    scale = 1
    if Max != 0:
        scale = torch.round(Limiter/Max)
    return (scale)

def QuantizeForShift(x):
    Max = np.max(abs(x))
    Q = 0
    if Max > 0:
        Q = np.log2((127/Max))
        if Q<0:
            Q = np.floor(Q)
        else:
            Q = np.round(Q)
        out = x*pow(2,Q)
        while np.max(out)>127 or np.min(out)<-128:
            Q = Q-1
            out = x*pow(2,Q)
    return(Q)

def QuantizeChannel(style, x):
    shape = x.shape
    if style == 'shift':
        power = []
        if len(shape) == 1:
            for i in range(x.shape[0]):
                Q = QuantizeForShift(x[i])
                power.append(Q)
        else:
            for i in range(x.shape[1]):
                Q = QuantizeForShift(x[:,i])
                power.append(Q)
        power = np.array(power)
        mean = power.mean()
        for i in range(power.shape[0]):
            if power[i] > 0:
                if power[i] > mean:
                    power[i] = math.floor(mean)
            if power[i] < 0:
                if power[i] < mean:
                    power[i] = math.ceil(mean)
        return (power)
    if style == 'simple' or 'winograd':
        scale_list = []
        if len(shape) == 1:
            for i in range(x.shape[0]):
                scale = QuantizeLinear(x[i], n)
                scale_list.append(scale)
        else:
            for i in range(x.shape[1]):
                scale = QuantizeLinear(x[i], n)
                scale_list.append(scale)
        return (scale_list)

    

if __name__ == "__main__":
    
    batch_size = 1
    # set up your network and quantization style, the network arg. format should be: 'googlenet', 'resnet50', 'squeezenet', 'ssd' ...
    # the quantization style should be -- shift, simple or winograd
    style = sys.argv[1]
    net_name = sys.argv[2]
    q_path = 'channel_q/' + net_name
    model, layer_name, Features, feature_path = load_model(net_name)
    layer_num = len(layer_name)
    print('Start quantization: ')
    print('Total layers is: ',end='')
    print(layer_num)
    for m in range(layer_num):
        print('layer: ',end='')
        print(m)
        feature_shape = Features[m].shape
        dim = len(feature_shape)
        if dim == 1:
            with open(feature_path + '/' + layer_name[m] + '.bin',"rb") as bin_reader:
                for i in range(feature_shape[0]):
                        data = bin_reader.read(4)
                        data_float = struct.unpack("f",data)[0]
                        Features[m][i] = data_float
        if dim == 2:
            with open(feature_path + '/' + layer_name[m] + '.bin',"rb") as bin_reader:
                for i in range(feature_shape[0]):
                    for j in range(feature_shape[1]):
                        data = bin_reader.read(4)
                        data_float = struct.unpack("f",data)[0]
                        Features[m][i][j] = data_float
        if dim == 3:
            with open(feature_path + '/' + layer_name[m] + '.bin',"rb") as bin_reader:
                for i in range(feature_shape[0]):
                    for j in range(feature_shape[1]):
                        for k in range(feature_shape[2]):
                            data = bin_reader.read(4)
                            data_float = struct.unpack("f",data)[0]
                            Features[m][i][j][k] = data_float
        if dim == 4:
            with open(feature_path + '/' + layer_name[m] + '.bin',"rb") as bin_reader:
                for i in range(feature_shape[0]):
                    for j in range(feature_shape[1]):
                        for k in range(feature_shape[2]):
                            for l in range(feature_shape[3]):
                                data = bin_reader.read(4)
                                data_float = struct.unpack("f",data)[0]
                                Features[m][i][j][k][l] = data_float
        power = QuantizeChannel(style, Features[m])
        if not os.path.exists(q_path):
            os.makedirs(q_path)
        with open(q_path + '/' + layer_name[m] + '.txt','w') as data:
            for item in power:
                item = int(item)
                item = str(item)
                item = item + ' '
                data.write(item)

