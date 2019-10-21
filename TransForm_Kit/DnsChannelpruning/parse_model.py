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
from torch.autograd import Variable
import os
import torch.nn as nn
class ParseModel():
    def __init__(self):
        self.visited = set()
        self.tops_dict = dict()
        self.layer_type_count = dict()
        self.slice_point = dict()
        self.multi_tops = dict()
        self.axis_dict = dict()

    def find_muti_tops(self,func):
        """
            Precount nodes with number of tops(indegree)>1,
            which could be Slice or Split(only in ncnn, for making multiple copies)
        """
        if func in self.visited:
            return self.tops_dict[func]

        self.visited.add(func)
        layer_type = str(type(func).__name__)
        bottoms = []

        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        child_name = self.find_muti_tops(u[0])
                        bottoms.append(child_name)

        """ Gen layer name """
        layer_type_name = layer_type.replace('Backward', '')
        if layer_type_name in self.layer_type_count:
            self.layer_type_count[layer_type_name] += 1
        else:
            self.layer_type_count[layer_type_name] = 1

        name = layer_type_name + '_' + str(self.layer_type_count[layer_type_name])

        """  Skip some pytorch layers  """
        #if dst == 'caffe':
        if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem']:
            self.tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            self.tops_dict[func] = bottoms[0]
        else:
            self.tops_dict[func] = name

        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        father_func = u[0]
                        if father_func not in self.multi_tops:
                            self.multi_tops[father_func] = []
                        self.multi_tops[father_func].append([self.tops_dict[father_func],self.tops_dict[func]])

                        if (layer_type == 'IndexBackward') and isinstance(func.index, tuple):
                            if father_func not in self.slice_point:
                                self.slice_point[father_func] = []
                            start, stop, dim_size, axis = GetLayerParam_Index(func)

                            """ Persume the visit of Index layers will be ascending """
                            if start > 0:
                                self.slice_point[father_func].append(start)
                                self.axis_dict[father_func] = axis

                                """ Last slice """
                                # if stop == dim_size

        return self.tops_dict[func]

    def parse_model_caffe(self,pytorch_net, InputShape, softmax = False):
        """  only support single tensor input """
        """ Need forward once """
        pytorch_net.eval()
        n, c, h, w = InputShape
        inputs = Variable(torch.rand(n, c, h, w), requires_grad=True)
        outputs = pytorch_net(inputs)

        if softmax:
            regularize = nn.Softmax()
            outputs = regularize(outputs)

        """ Travel computational graph in backward order """
        """ Need to count number of tops(indegree) of all nodes first """
        for out in outputs:
            self.find_muti_tops(out.grad_fn)
        
        layername_type_dict = {}
        layername_lianjie_dict = {}
        for k,v in self.multi_tops.items():
            layername = v[0][0]
            layername_type_dict[layername] = str(type(k).__name__).replace('Backward', '')
            layername_lianjie_dict[layername] = []
            for v_value in v:
                layername_lianjie_dict[layername].append(v_value[1])
        for k,v in layername_type_dict.items():
            print(k,v,layername_lianjie_dict[k],len(layername_lianjie_dict[k]))
        print("layername type dict")
        for k,v in layername_type_dict.items():
            print(k,v)
        return layername_type_dict,layername_lianjie_dict