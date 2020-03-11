"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import math
import numpy as np
import caffe_pb2 as pb2
import torch.nn as nn

class networkbuild():
    def __init__(self):
        self._convindex = 0
        self._bnindex = 0
        self._fcindex = 0
        self._poolindex = 0
        self._conv_module_dict = {}
        self._bn_module_dict = {}
        self._fc_module_dict ={}
        self._pool_module_dict = {}
    def module_to_dict(self,pytorch_net):
        conv_index = 0
        bn_index = 0
        fc_index = 0
        pool_index = 0
        for module in pytorch_net.modules():
            if isinstance(module,nn.Conv2d):
                self._conv_module_dict[conv_index] = module
                conv_index += 1
            if isinstance(module,nn.BatchNorm2d):
                self._bn_module_dict[bn_index] = module
                bn_index += 1
            if isinstance(module,nn.Linear):
                self._fc_module_dict[fc_index] = module
                fc_index += 1
            if isinstance(module,nn.MaxPool2d) or isinstance(module,nn.AvgPool2d):
                self._pool_module_dict[pool_index] = module
                pool_index += 1

    def as_blob(self,array):
        blob = pb2.BlobProto()
        blob.shape.dim.extend(array.shape)
        blob.data.extend(array.astype(float).flat)
        return blob


    def CopyTuple(self,param):
        if isinstance(param, tuple):
            return param
        elif isinstance(param, int):
            return param, param
        else:
            assert type(param)


    def ty(self,caffe_type):
        def f(_):
            layer = pb2.LayerParameter()
            layer.type = caffe_type
            return layer
        return f


    def data(self,inputs):
        layer = pb2.LayerParameter()
        layer.type = 'Input'
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(inputs.data.numpy().shape)
        layer.input_param.shape.extend([input_shape])
        return layer


    def Slice(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        layer.slice_param.axis = pytorch_layer.axis
        layer.slice_param.slice_point.extend(pytorch_layer.slice_point)
        return layer


    def inner_product(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"
        fcmodule = self._fc_module_dict[self._fcindex]
        blobs_weight = fcmodule.weight.data.cpu().numpy()
        num_output = blobs_weight.shape[0]
        layer.inner_product_param.num_output = num_output
        print(blobs_weight.shape)
        if fcmodule.bias is not None:
            layer.inner_product_param.bias_term = True
            bias = fcmodule.bias.data.cpu().numpy()
            print(bias.shape)
            layer.blobs.extend([self.as_blob(blobs_weight), self.as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([self.as_blob(blobs_weight)])
        self._fcindex += 1
        return layer


    def concat(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Concat"
        layer.concat_param.axis = int(pytorch_layer.dim)
        return layer


    def flatten(self,pytorch_layer):
        """ Only support flatten view """
        total = 1
        old_size = [2048,1,1] #pytorch_layer.old_size
        for dim in old_size:
            total *= dim
        #assert ((pytorch_layer.new_sizes[1] == total) or (pytorch_layer.new_sizes[1] == -1))

        layer = pb2.LayerParameter()
        layer.type = "Flatten"
        return layer


    def spatial_convolution(self,pytorch_layer):
        layer = pb2.LayerParameter()
        convmodel = self._conv_module_dict[self._convindex]
        blobs_weight = convmodel.weight.data.cpu().numpy()
        #blobs_weight = pytorch_layer.next_functions[1][0].variable.data.numpy()
        #blobs_weight = convmodel.
        assert len(blobs_weight.shape) == 4, blobs_weight.shape
        (nOutputPlane, nInputPlane, kH, kW) = blobs_weight.shape
        padH =  convmodel.padding[0]#pytorch_layer.padding[0]
        padW =  convmodel.padding[1] #pytorch_layer.padding[1]
        dH = convmodel.stride[0] #pytorch_layer.stride[0]
        dW = convmodel.stride[1] #pytorch_layer.stride[1]
        dilation = convmodel.dilation[0] #pytorch_layer.dilation[0]
        transposed = convmodel.transposed  #pytorch_layer.transposed 
        groups = convmodel.groups #pytorch_layer.groups
        #print(blobs_weight)
        #print(convmodel.weight.data)
        if transposed:
            layer.type = "Deconvolution"
            layer.convolution_param.num_output = nInputPlane
        else:
            layer.type = "Convolution"
            layer.convolution_param.num_output = nOutputPlane

        if kH == kW:
            layer.convolution_param.kernel_size.extend([kH])
        else:
            layer.convolution_param.kernel_h = kH
            layer.convolution_param.kernel_w = kW
        if dH == dW:
            layer.convolution_param.stride.extend([dH])
        else:
            layer.convolution_param.stride_h = dH
            layer.convolution_param.stride_w = dW
        if padH == padW:
            layer.convolution_param.pad.extend([padH])
        else:
            layer.convolution_param.pad_h = padH
            layer.convolution_param.pad_w = padW
        layer.convolution_param.dilation.extend([dilation])
        layer.convolution_param.group = groups

        if convmodel.bias:
            layer.convolution_param.bias_term = True
            bias = convmodel.bias.data
            layer.blobs.extend([self.as_blob(blobs_weight), self.as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([self.as_blob(blobs_weight)])
        print('convindex:',self._convindex)
        self._convindex += 1
        return layer


    def FillBilinear(self,ch, k):
        blob = np.zeros(shape=(ch, 1, k, k))

        """ Create bilinear weights in numpy array """
        bilinear_kernel = np.zeros([k, k], dtype=np.float32)
        scale_factor = (k + 1) // 2
        if k % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(k):
            for y in range(k):
                bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)

        for i in range(ch):
            blob[i, 0, :, :] = bilinear_kernel
        return blob


    def UpsampleBilinear(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Deconvolution"

        assert pytorch_layer.scale_factor[0] == pytorch_layer.scale_factor[1]
        factor = int(pytorch_layer.scale_factor[0])
        c = int(pytorch_layer.input_size[1])
        k = 2 * factor - factor % 2

        layer.convolution_param.num_output = c
        layer.convolution_param.kernel_size.extend([k])
        layer.convolution_param.stride.extend([factor])
        layer.convolution_param.pad.extend([int(math.ceil((factor - 1) / 2.))])
        layer.convolution_param.group = c
        layer.convolution_param.weight_filler.type = 'bilinear'
        layer.convolution_param.bias_term = False

        learning_param = pb2.ParamSpec()
        learning_param.lr_mult = 0
        learning_param.decay_mult = 0
        layer.param.extend([learning_param])

        """ Init weight blob of filter kernel """
        blobs_weight = FillBilinear(c, k)
        layer.blobs.extend([self.as_blob(blobs_weight)])

        return layer


    def CopyPoolingParameter(self,pytorch_layer, layer):

        poolmodule = self._pool_module_dict[self._poolindex]
        self._poolindex += 1
        kH, kW = self.CopyTuple(poolmodule.kernel_size)
        dH, dW = self.CopyTuple(poolmodule.stride)
        padH, padW = self.CopyTuple(poolmodule.padding)
        #kH, kW = self.CopyTuple(0)
        #dH, dW = self.CopyTuple(0)
        #padH, padW = self.CopyTuple(0)
        ceil_mode = poolmodule.ceil_mode  
        if kH == kW:
            layer.pooling_param.kernel_size = kH
        else:
            layer.pooling_param.kernel_h = kH
            layer.pooling_param.kernel_w = kW
        if dH == dW:
            layer.pooling_param.stride = dH
        else:
            layer.pooling_param.stride_h = dH
            layer.pooling_param.stride_w = dW
        if padH == padW:
            layer.pooling_param.pad = padH
        else:
            layer.pooling_param.pad_h = padH
            layer.pooling_param.pad_w = padW

        if ceil_mode is True:
            return

        if ceil_mode is False:
            if padH == padW:
                if dH > 1 and padH > 0:
                    layer.pooling_param.pad = padH - 1
            else:
                if dH > 1 and padH > 0:
                    layer.pooling_param.pad_h = padH - 1
                if dW > 1 and padW > 0:
                    layer.pooling_param.pad_w = padW - 1
        

    def MaxPooling(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"
        layer.pooling_param.pool = pb2.PoolingParameter.MAX
        self.CopyPoolingParameter(pytorch_layer, layer)
        return layer


    def AvgPooling(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"
        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        self.CopyPoolingParameter(pytorch_layer, layer)
        return layer


    def dropout(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Dropout"
        layer.dropout_param.dropout_ratio = float(pytorch_layer.p)
        train_only = pb2.NetStateRule()
        train_only.phase = pb2.TEST
        layer.exclude.extend([train_only])
        return layer


    def elu(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "ELU"
        layer.elu_param.alpha = pytorch_layer.additional_args[0]
        return layer


    def leaky_ReLU(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"
        layer.relu_param.negative_slope = float(pytorch_layer.additional_args[0])
        return layer


    def PReLU(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "PReLU"
        num_parameters = int(pytorch_layer.num_parameters)
        layer.prelu_param.channel_shared = True if num_parameters == 1 else False

        blobs_weight = pytorch_layer.next_functions[1][0].variable.data.numpy()
        layer.blobs.extend([self.as_blob(blobs_weight)])
        return layer


    def MulConst(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Power"
        layer.power_param.power = 1
        layer.power_param.scale = float(pytorch_layer.constant)
        layer.power_param.shift = 0
        return layer


    def AddConst(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Power"
        layer.power_param.power = 1
        layer.power_param.scale = 1
        """ Constant to add should be filled by hand, since not visible in autograd """
        layer.power_param.shift = float('inf')
        return layer


    def softmax(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = 'Softmax'
        return layer


    def eltwise(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"
        return layer


    def eltwise_max(self,pytorch_layer):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"
        layer.eltwise_param.operation = 2
        return layer


    def batchnorm(self,pytorch_layer):
        layer_bn = pb2.LayerParameter()
        layer_bn.type = "BatchNorm"
        bnmodule = self._bn_module_dict[self._bnindex]
        layer_bn.batch_norm_param.use_global_stats = 1
        layer_bn.batch_norm_param.eps = bnmodule.eps #pytorch_layer.eps
        print(type(bnmodule.eps))
        layer_bn.blobs.extend([
            self.as_blob(bnmodule.running_mean.cpu().numpy()),
            self.as_blob(bnmodule.running_var.cpu().numpy()),
            #self.as_blob(np.array([1.]))
        ])
        #layer_bn.blobs.extend([
            #self.as_blob(bnmodule.weight.data.cpu().numpy()),
            #self.as_blob(bnmodule.weight.data.cpu().numpy()),
            #self.as_blob(np.array([1.]))
        #])
        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"

        blobs_weight = bnmodule.weight.data.cpu().numpy()

        if bnmodule.bias is not None:
            layer_scale.scale_param.bias_term = True
            bias = bnmodule.bias.data.cpu().numpy()
            layer_scale.blobs.extend([self.as_blob(blobs_weight), self.as_blob(bias)])
        else:
            layer_scale.scale_param.bias_term = False
            layer_scale.blobs.extend([self.as_blob(blobs_weight)])
        self._bnindex += 1
        return [layer_bn, layer_scale]


    def build_converter(self,opts):
        return {
            'data': self.data,
            'ThAddmm': self.inner_product,
            'Addmm': self.inner_product,
            'Threshold1': self.ty('ReLU'),
            'Relu1': self.ty('ReLU'),
            'ThnnConv2D': self.spatial_convolution,
            'CudnnConvolution': self.spatial_convolution,
            'MaxPool2DWithIndices': self.MaxPooling,
            'AvgPool2D': self.AvgPooling,
            'Mean1': self.AvgPooling,
            'ThAdd': self.eltwise,
            'Add0': self.eltwise,
            'Cmax': self.eltwise_max,
            'ThnnBatchNorm': self.batchnorm,
            'CudnnBatchNorm': self.batchnorm,
            'Concat': self.concat,
            'Dropout': self.dropout,
            'UpsamplingBilinear2d': self.UpsampleBilinear,
            'MulConstant': self.MulConst,
            'AddConstant': self.AddConst,
            'Select': self.softmax,
            'Sigmoid': self.ty('Sigmoid'),
            'Tanh': self.ty('TanH'),
            'ELU': self.elu,
            'LeakyReLU': self.leaky_ReLU,
            'PReLU': self.PReLU,
            'Slice': self.Slice,
            'View': self.flatten,
        }


    def convert_caffe(self,opts, typename, pytorch_layer):
        converter = self.build_converter(opts)
        if typename not in converter:
            raise ValueError("Unknown layer type: {}, known types: {}".format(
                typename, converter.keys()))
        return converter[typename](pytorch_layer) 
