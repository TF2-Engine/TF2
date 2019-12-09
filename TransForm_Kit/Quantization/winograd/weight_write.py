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

import numpy as np
import struct

def LinearQuantization(x, n):
    Max = torch.max(abs(x))
    Limiter = pow(2,n-1) - 1
    scale = 1
    if Max != 0:
        x = torch.round(x*(Limiter/Max))
        scale = torch.round(Limiter/Max)
    return (x, scale)

def QuantizeChannel(x, n):
    scale = []
    for i in range(x.shape[0]):
        x[i], int8_scale = LinearQuantization(x[i], n)
        scale.append(int8_scale)
    return x, scale

def WeightWrite(net_name, x):
    with open(net_name + '.bin','ab+') as data:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        st = x[i][j][k][l].item()
                        st = struct.pack('f',st)
                        data.write(st)

def WeightWriteScale(net_name, x):
    with open(net_name + '.bin','ab+') as data:
        st = x.item()
        st = struct.pack('f',st)
        data.write(st)

def WeightWriteFC(net_name, x):
    with open(net_name + '.bin','ab+') as data:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                st = x[i][j].item()
                st = struct.pack('f',st)
                data.write(st)

def BiasWriteFC(net_name, x):
    with open(net_name + '.bin','ab+') as data:
        for i in range(x.shape[0]):
            st = x[i].item()
            st = struct.pack('f',st)
            data.write(st)

