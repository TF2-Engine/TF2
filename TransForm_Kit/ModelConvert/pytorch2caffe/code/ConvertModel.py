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
import caffe_pb2 as pb2
from ConvertLayer_caffe import networkbuild

def link_caffe(layers, name, bottom, top):
    layers.name = name
    for b in bottom:
        if layers.type == "InnerProduct":
            if b == "data":
                continue
        '''if b.find('Broadcast')>=0 or b.find('Scatter')>=0:
            continue'''
        layers.bottom.append(b)
    for t in top:
        layers.top.append(t)
    caffe_net.append(layers)

def GetLayerParam_Index(func):
    for axis, slice_param in enumerate(func.index):
        if isinstance(slice_param, int):
            start = slice_param
            stop = slice_param + 1
        else:
            start = slice_param.start
            stop = slice_param.stop
            step = slice_param.step
        if (start or stop or step) is not None:
            break
    shape = func.input_size
    dim_size = shape[axis]
    return start, stop, dim_size, axis


def DFS(func):
    global bottomlayer_top
    if func in visited:
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    father_func = None
    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:                                     
                child_type = str(type(u[0]).__name__)
                print('child_type:',child_type,'layer type:',layer_type)
                if child_type != 'AccumulateGrad' and (layer_type != 'ThAddmmBackward' or child_type != 'TransposeBackward'):
                    print('digui:',layer_type.replace('Backward', ''))
                    child_name = DFS(u[0])
                    if (child_name.find('Broadcast')<0 and child_name.find('Scatter')<0 \
                        and child_name.find('Gather')<0) or layer_type == 'GatherBackward':
                        bottoms.append(child_name)
                        father_func = u[0]
                        print('bottoms:',bottoms)

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1
    print('layer type name :',layer_type_name)
    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """ Reaching the root node """
    """  TODO: multi data input """
    if len(bottoms) == 0:
        if 'data' not in layer_type_count:
            layer_type_count['data'] = 1
            """ Gen data layer """
            layer_data = convert('', 'data', inputs)
            link(layer_data, 'data', [], ['data'])

        """ Link it with data input """
        if layer_type_name == 'Select':
            bottoms = bottomlayer_top
        else:
            bottoms.append('data')

    """  Skip some pytorch layers  """
    
    if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem','Expand','T','View']:
        tops_dict[func] = bottoms[0]
    elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
        tops_dict[func] = bottoms[0]
    else:
        tops_dict[func] = name
        if layer_type_name == 'Index':
            """ Change layer name only for 'Slice' """
            tops_dict[func] = tops_dict[father_func] + '_' + tops_dict[func]
    
    """ Split to BatchNorm and Scale """
    if layer_type_name == 'ThnnBatchNorm' or layer_type_name == 'CudnnBatchNorm':
        print('convert:',layer_type_name)
        layer_double = convert('', layer_type_name, func)
        scale_name = name + '_' + 'scale'
        
        link(layer_double[0], name, bottoms, [tops_dict[func]])
        link(layer_double[1], scale_name, [tops_dict[func]], [tops_dict[func]])
        bottomlayer_top = [tops_dict[func]]
    elif layer_type_name not in ['Index', 'Clone', 'SetItem','Expand','T','View','Broadcast','Scatter','Gather']:
            """ Debug """
            # if layer_type_name != 'Cmax':
            #     return tops_dict[func]
            print('convert:',layer_type_name)
            layer = convert('', layer_type_name, func)
            link(layer, name, bottoms, [tops_dict[func]])
            bottomlayer_top = [tops_dict[func]]
    """ If func layer has multiple top layers """
    if (func in multi_tops) and (len(multi_tops[func]) > 1):
        if func in slice_point:
            """ Make an extra dummy layer type 'Slice' after func layer, which not exist in pytorch """
            slice_func = torch.autograd.function
            slice_func.axis = axis_dict[func]
            slice_func.slice_point = slice_point[func]
            slice_layer = convert('', 'Slice', slice_func)
            link(slice_layer, tops_dict[func] + '_slicer', [tops_dict[func]], multi_tops[func])
            bottomlayer_top = multi_tops[func]
    return tops_dict[func]


def FindMultiTops(func):
    """
        Precount nodes with number of tops(indegree)>1,
        which could be Slice or Split(only in ncnn, for making multiple copies)
    """
    if func in visited:
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    child_name = FindMultiTops(u[0])
                    bottoms.append(child_name)

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """  Skip some pytorch layers  """
    
    if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem']:
        tops_dict[func] = bottoms[0]
    elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
        tops_dict[func] = bottoms[0]
    else:
        tops_dict[func] = name
    

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                    father_func = u[0]
                    if father_func not in multi_tops:
                        multi_tops[father_func] = []
                    multi_tops[father_func].append(tops_dict[father_func] + '_' + tops_dict[func])

                    if (layer_type == 'IndexBackward') and isinstance(func.index, tuple):
                        if father_func not in slice_point:
                            slice_point[father_func] = []
                        start, stop, dim_size, axis = GetLayerParam_Index(func)

                        """ Persume the visit of Index layers will be ascending """
                        if start > 0:
                            slice_point[father_func].append(start)
                            axis_dict[father_func] = axis

                            """ Last slice """
                            # if stop == dim_size

    return tops_dict[func]

def ConvertModel_caffe(pytorch_net, InputShape, softmax=False):
    """ Pytorch to Caffe, only support single tensor input """
    
    #from ConvertLayer_caffe import convert_caffe
    """ Need forward once """
    pytorch_net.eval()
    global inputs
    n, c, h, w = InputShape
    inputs = Variable(torch.rand(n, c, h, w), requires_grad=True)
    outputs = pytorch_net(inputs)

    if softmax:
        import torch.nn as nn
        regularize = nn.Softmax()
        outputs = regularize(outputs)

    """ Travel computational graph in backward order """
    """ Need to count number of tops(indegree) of all nodes first """
    global visited, tops_dict, layer_type_count, dst
    global slice_point, multi_tops, axis_dict,bottomlayer_top
    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    slice_point = dict()
    multi_tops = dict()
    axis_dict = dict()
    
    for out in outputs:
        FindMultiTops(out.grad_fn)
    for k,v in tops_dict.items():
        print(k,v)
    print("slice_point dict")
    for k,v in slice_point.items():
        print(k,v)
    print("multi_tops dict")
    for k,v in multi_tops.items():
        print(k,v)
    print("axis_dict")
    for k,v in axis_dict.items():
        print(k,v)
    print("layer_type_count")
    for k,v in layer_type_count.items():
        print(k,v)
    """ Travel computational graph in backward order """
    global caffe_net
    global convert, link
    caffe_networkbuild = networkbuild()
    caffe_networkbuild.module_to_dict(pytorch_net)
    convert = caffe_networkbuild.convert_caffe
    link = link_caffe
    caffe_net = []

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()

    for out in outputs:
        DFS(out.grad_fn)

    """ Caffe input """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True

    """ Caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for layer in caffe_net:
        binary_weights.layer.extend([layer])

        layer_proto = pb2.LayerParameter()
        layer_proto.CopyFrom(layer)
        del layer_proto.blobs[:]
        text_net.layer.extend([layer_proto])

    return text_net, binary_weights
