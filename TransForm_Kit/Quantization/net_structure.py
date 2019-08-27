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
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
def structure_hook(model,net_name):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(layers)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            layers.append(m_key)
            shape = list(output.size())
            dim = len(shape)
            print(shape)
            if dim == 1:
                fea = np.zeros((shape[0]))
            if dim == 2:
                fea = np.zeros((shape[0],shape[1]))
            if dim == 3:
                fea = np.zeros((shape[0],shape[1],shape[2]))
            if dim == 4:
                fea = np.zeros((shape[0],shape[1],shape[2],shape[3]))
            feas.append(fea)
        hooks.append(module.register_forward_hook(hook))
    layers = []
    feas = []
    hooks = []
    model.apply(register_hook)
    if net_name == 'squeezenet':
        x = torch.zeros(1,3,160,160)
    if net_name == 'ssd':
        x = torch.zeros(1,3,300,300)
    if net_name == 'resnet50' or net_name == 'googlenet':
        x = torch.zeros(1,3,224,224)
    model(x)
    for h in hooks:
        h.remove()
    return layers, feas
