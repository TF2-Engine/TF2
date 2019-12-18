

import torch
import cv2
import numpy as np
import struct
from PIL import Image
import math
import numpy as np
from numpy import mat
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import os
import multiprocessing as mp
import sys
sys.path.append("..")
from conv7x7_2_3x3 import Conv1FeatureTransform
from conv7x7_2_3x3 import Conv1FilterTransform
from filter_transform import FilterTransform
from feature_transform import FeatureTransform
from weight_write import QuantizeChannel
from weight_write import LinearQuantization

def Conv1_7x7_2_3x3(x, Filter):
    feature_list = Conv1FeatureTransform(x)
    weight_list = Conv1FilterTransform(Filter)
    for i in range(9):
        feature_list[i] = torch.Tensor(feature_list[i])
        #----winograd------
        filter_shape = weight_list[i].shape
        filter_trans = np.zeros((filter_shape[0],filter_shape[1],3,7))
        filter_trans = FilterTransform(filter_shape,weight_list[i],filter_trans,5,1)
        filter_trans = np.float32(filter_trans)
        filter_trans = np.round(filter_trans)
        filter_trans[filter_trans>32767] = 32767
        filter_trans[filter_trans<-32768] = -32768
        x = Conv2dWinograd(feature_list[i],filter_trans)
        if 'output' in dir():
            output = output + x
        else:
            output = x

    # AT(~)  transverse the sum result with the AT matrix
    AT_list = [1,1,1,1,1,1,0,0,1,-1,2,-2,3,0,0,1,1,4,4,9,0,0,1,-1,8,-8,27,0,0,1,1,16,16,81,1]
    AT = np.zeros((5,7))
    AT_count = 0
    for i in range(5):
        for j in range(7):
            AT[i,j] = AT_list[AT_count]
            AT_count = AT_count + 1
    AT = mat(AT)
    conv_result = np.zeros((1,64,112,int(5*output.shape[3]/7)))
    column_minus = 3
    conv_w_count = 0
    conv_w_start = 0
    for i in range(1):
        for j in range(64):
            for k in np.arange(112)[::1]:
                for l in np.arange(output.shape[3])[::7]:
                    mid_result = output[i,j,k//1,l:l+7]   
                    mid_result = mat(mid_result)
                    out_mid = AT*(mid_result.T)
                    conv_result[i,j,k//1,conv_w_start:conv_w_start+5] = out_mid.T
                    conv_w_count = conv_w_count + 1
                    conv_w_start = 5*conv_w_count
                conv_w_count = 0
                conv_w_start = 0
    last_result = conv_result[0:1, :, :, :conv_result.shape[3]-column_minus]
    return last_result


def Conv2dWinograd(x, weight):
    
    weight = np.array(weight)
    shape = x.shape
    if shape[2] == 114:
        padding = 3
        column_minus = 3
        npd = ((0,0),(0,0),(0,0),(0,padding))
    if shape[2] == 56:
        padding = 6
        column_minus = 4
        npd = ((0,0),(0,0),(1,1),(1,padding-1))
    if shape[2] == 28:
        padding = 4
        column_minus = 2
        npd = ((0,0),(0,0),(1,1),(1,padding-1))
    if shape[2] == 14:
        padding = 3
        column_minus = 1
        npd = ((0,0),(0,0),(1,1),(1,padding-1))
    if shape[2] == 7:
        padding = 5
        column_minus = 3
        npd = ((0,0),(0,0),(1,1),(1,padding-1))
    x = np.lib.pad(x, npd, 'constant', constant_values=0)
    expand_x = np.zeros((shape[0],shape[1],x.shape[2],int(7*(x.shape[3]-2)/5)))
    expand_x = np.float32(expand_x)
    expand_w_count = 0
    expand_w_start = 0
    for k in range(x.shape[2]):
        for l in np.arange(shape[3])[::5]:
            expand_x[:,:,k:k+1,expand_w_start:expand_w_start+7] = FeatureTransform(shape, x[:,:,k:k+1,l:l+7],5,1)
            expand_w_count = expand_w_count + 1
            expand_w_start = 7*expand_w_count
        expand_w_count = 0
        expand_w_start = 0
   
    expand_x = np.round(expand_x)
    expand_x[expand_x>32767] = 32767
    expand_x[expand_x<-32768] = -32768
    AT_list = [1,1,1,1,1,1,0,0,1,-1,2,-2,3,0,0,1,1,4,4,9,0,0,1,-1,8,-8,27,0,0,1,1,16,16,81,1]
    AT = np.zeros((5,7))
    AT_count = 0
    for i in range(5):
        for j in range(7):
            AT[i,j] = AT_list[AT_count]
            AT_count = AT_count + 1
    AT = mat(AT) 
    out_h = int((x.shape[2] - weight.shape[2])/1 + 1) #stride = 1
    out_w = int(5*expand_x.shape[3]/7)
    conv_result = np.zeros((1, weight.shape[0], out_h, out_w))
    mid_out_result = np.zeros((1, weight.shape[0], out_h, expand_x.shape[3])) 
    conv_w_count = 0
    conv_w_start = 0
    if shape[2] != 114:
        for i in range(1):
            for j in range(weight.shape[0]):
                for k in np.arange(x.shape[2] - weight.shape[2] + 1)[::1]:
                    for l in np.arange(expand_x.shape[3])[::7]:
                        mid_result = np.sum(expand_x[0, :, k:k+weight.shape[2], l:l+7]*weight[j,:], axis=0)
                        mid_result = np.sum(mid_result, axis=0)
                        mid_result = mat(mid_result)
                        out_mid = AT*(mid_result.T)
                        conv_result[i,j,k//1,conv_w_start:conv_w_start+5] = out_mid.T   
                        conv_w_count = conv_w_count + 1
                        conv_w_start = 5*conv_w_count
                    conv_w_count = 0
                    conv_w_start = 0
        last_result = conv_result[0:1, :, :, :conv_result.shape[3]-column_minus]
    
    if shape[2] == 114:
        for i in range(1):
            for j in range(weight.shape[0]):
                for k in np.arange(x.shape[2] - weight.shape[2] + 1)[::1]:
                    for l in np.arange(expand_x.shape[3])[::7]:
                        mid_result = np.sum(expand_x[0, :, k:k+weight.shape[2], l:l+7]*weight[j,:], axis=0)
                        mid_result = np.sum(mid_result, axis=0)
                        mid_out_result[i,j,k//1,l:l+7] = mid_result
        last_result = mid_out_result[0:1, :, :, :]
    return last_result

def BN(conv_result,alpha,beta,scale_filter,scale_out):
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    N = conv_result.shape[0]
    out_c = conv_result.shape[1]
    out_h = conv_result.shape[2]
    out_w = conv_result.shape[3]
    bn_result = np.zeros((N,out_c,out_h,out_w))
    bn_result = np.float32(bn_result)
    for i in range(N):
        for j in range(out_c):
            alpha_out = np.round(alpha[j]*conv_result[i][j])
            beta_out = np.round(beta[j]*scale_out)
            bn_result[i][j] = alpha_out + beta_out
    bn_result = np.round(bn_result)
    bn_result[bn_result>127] = 127
    bn_result[bn_result<-128] = -128
    return (bn_result)

def FilterCounter(last=[-1]):
    filter_count = last[0] + 1
    last[0] = filter_count
    return filter_count

class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000):#True-original
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        global filter_name, Filter, fc_bias, scale_feature, scale_dict#, filter_count
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        filter_count = FilterCounter()
        filter_key = filter_name[filter_count]
        x = F.linear(x, Filter[filter_key], fc_bias)
        x = x/(scale_feature[filter_count]*scale_dict[filter_key])
        FilterCounter(last=[-2])
        return x

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, stride=1, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): # **kwargs
        super(BasicConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
    def forward(self, x):
        global wino_count, filter_name, Filter, bn_name, running_mean, running_var, bn_weight, bn_bias, scale_feature, scale_dict#, filter_count
        padding = self.padding
        stride = self.stride
        filter_count = FilterCounter()
        filter_key = filter_name[filter_count]
        bn_key = bn_name[filter_count]
        if (filter_count-5)%6 == 0 or (filter_count-7)%6 == 0 or filter_count == 2:
            if filter_count != 1:
                x = Conv2dWinograd(x, Filter[filter_key])
                x = torch.Tensor(x)
                x = x/(120*scale_dict[filter_key])
                wino_count = wino_count + 1
        if filter_count != 0 and filter_count != 2 and (filter_count-7)%6 != 0 and (filter_count-5)%6 != 0:
            x = F.conv2d(x, Filter[filter_key], stride=stride, padding=padding)
            x = x/scale_dict[filter_key]
        if filter_count == 1: # or filter_count == 0:
            x = F.conv2d(x, Filter[filter_key], stride=stride, padding=padding)
            x = x/scale_dict[filter_key]
        if filter_count == 0:
            x = Conv1_7x7_2_3x3(x,Filter[filter_key])
            x = np.float32(x)
            x = torch.Tensor(x)
            x = x/(120*scale_dict[filter_key])
        x = torch.Tensor(x)
        x = torch.round(x)
        x = BN(x,alpha[bn_key],beta[bn_key],scale_dict[filter_key],scale_feature[filter_count+1])
        x = torch.Tensor(x)
        return F.relu(x, inplace=True)

class FurtherProcess(object):
    def __call__(self,x):
        x[0] = x[0]* (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[1] = x[1]* (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[2] = x[2]* (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return x

def accuracy(output,target,topk=(1,5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def net_inference(proc_th, n, Temp1, Temp5):
    global val, proc_num, scale_feature
    cnn = GoogLeNet()
    cnn.eval()
    test_x = torch.zeros(1,3,224,224)
    test_y = torch.LongTensor(1)
    test_x[0] = val[n*proc_num + proc_th][0]
    test_y[0] = val[n*proc_num + proc_th][1]
    test_x = scale_feature[0]*test_x
    test_x = torch.round(test_x)
    test_x[test_x>127] = 127
    test_x[test_x<-128] = -128
    output_test = cnn(test_x)
    prec1,prec5 = accuracy(output_test.data,test_y,topk=(1,5))
    prec1 = prec1.cpu().numpy()
    prec5 = prec5.cpu().numpy()
    Temp1.append(prec1[0])
    Temp5.append(prec5[0])

def multicore(n, Temp1, Temp5):
    global proc_num
    pool = mp.Pool(processes=proc_num)
    for i in range(proc_num):
        pool.apply_async(net_inference,(i,n,Temp1,Temp5))
    pool.close()
    pool.join()

if __name__ == "__main__":

    batch_size = 1
    data_dir = './'
    import torchvision.datasets as datasets
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform = transforms.Compose([preprocess,FurtherProcess()])
    val = datasets.ImageFolder(os.path.join(data_dir,'val'),transform)
    ValLoader = data.DataLoader(val,batch_size=batch_size)

    pars = torch.load('googlenet.pth')
    filter_name = []
    bn_name = []
    Filter = OrderedDict()
    bn_weight = OrderedDict()
    bn_bias = OrderedDict()
    running_mean = OrderedDict()
    running_var = OrderedDict()
    alpha = OrderedDict()
    beta = OrderedDict()
    scale_dict = OrderedDict()

    layer_count = 0
    filter_count = 0
    count = 0

    for key in pars:
        filter_shape = pars[key].shape
        if count<343:
            if count%6 == 0:
                pars[key] = torch.Tensor(pars[key])
                pars[key], scale = LinearQuantization(pars[key], 8)
                pars[key][pars[key]>127] = 127
                pars[key][pars[key]<-128] = -128
                scale_dict[key] = scale
                if count == 342:
                    print(pars[key][999])
                if (filter_count-5)%6 == 0 or (filter_count-7)%6 == 0 or filter_count == 2:
                    if filter_count != 1:
                        filter_name.append(key)
                        pars[key] = pars[key].cpu().numpy()
                        filter_trans = np.zeros((filter_shape[0],filter_shape[1],3,7))
                        filter_trans = FilterTransform(filter_shape,pars[key],filter_trans,5,1)
                        Filter[key] = filter_trans
                if filter_count != 0 and filter_count != 2 and (filter_count-7)%6 != 0 and (filter_count-5)%6 != 0:
                    filter_name.append(key)
                    Filter[key] = pars[key]
                if filter_count == 1 or filter_count == 0:
                    filter_name.append(key)
                    Filter[key] = pars[key]
                Filter[key] = torch.Tensor(Filter[key])
                Filter[key] = torch.round(Filter[key])
                Filter[key][Filter[key]>32767] = 32767
                Filter[key][Filter[key]<-32768] = -32768
                filter_count = filter_count + 1
            if (count-1)%6 == 0:
                alpha0 = pars[key]
            if (count-2)%6 == 0:
                beta0 = pars[key]
            if (count-3)%6 == 0:
                mean = pars[key]
            if (count-4)%6 == 0:
                bn_name.append(key)
                bn_weight[key] = alpha0
                bn_bias[key] = beta0
                running_mean[key] = mean
                running_var[key] = pars[key]
                var = pars[key]
                alpha[key] = alpha0/(torch.sqrt(var+0.001))
                beta[key] = -alpha[key]*mean + beta0
                alpha[key] = alpha[key].cpu().numpy()
                beta[key] = beta[key].cpu().numpy()
            
        if count == 343:
            fc_bias = pars[key]
            print(fc_bias)
        count = count + 1

    scale_feature = torch.load('scale_feature_layer.pkl')
    scale_feature = torch.Tensor(scale_feature)
    filter_count = 0
    wino_count = 0
    total_pic = 1
    proc_num = 1
    loop_num = int(total_pic/proc_num)
    with mp.Manager() as mg:
        Temp1 = mg.list()
        Temp5 = mg.list()
        for i in range(loop_num):
            multicore(i, Temp1, Temp5)
            print(i)
        Top1 = np.array(Temp1)
        Top5 = np.array(Temp5)
        print('')
        print('')
        print("----------------END---------------")
        print('Top1 = ',end=' ')
        print(Top1.mean(),end=' ')
        print('Top5 = ',end=' ')
        print(Top5.mean())
