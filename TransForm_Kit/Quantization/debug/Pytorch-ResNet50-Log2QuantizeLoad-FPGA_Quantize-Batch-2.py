
# coding: utf-8


import math
import struct
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torchvision
import torchvision.models as tvmodel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import gc
#os.environ['CUDA_VISIBLE_DEVICES']='2'

layer_name_binQ = ['data','conv1-scale','res2a_branch2a-scale','res2a_branch2b-scale','res2c','res2b_branch2a-scale','res2b_branch2b-scale','res2c','res2c_branch2a-scale','res2c_branch2b-scale','res2c','res3a_branch2a-scale','res3a_branch2b-scale','res3d','res3b_branch2a-scale','res3b_branch2b-scale','res3d','res3c_branch2a-scale','res3c_branch2b-scale','res3d','res3d_branch2a-scale','res3d_branch2b-scale','res3d','res4a_branch2a-scale','res4a_branch2b-scale','res4f','res4b_branch2a-scale','res4b_branch2b-scale','res4f','res4c_branch2a-scale','res4c_branch2b-scale','res4f','res4d_branch2a-scale','res4d_branch2b-scale','res4f','res4e_branch2a-scale','res4e_branch2b-scale','res4f','res4f_branch2a-scale','res4f_branch2b-scale','res4f','res5a_branch2a-scale','res5a_branch2b-scale','res5c','res5b_branch2a-scale','res5b_branch2b-scale','res5c','res5c_branch2a-scale','res5c_branch2b-scale','res5c','fc1000']

layer_name_bin = ['data','conv1-scale','res2a_branch2a-scale','res2a_branch2b-scale','res2a_branch2c-scale','res2a_branch1-scale','res2a','res2b_branch2a-scale','res2b_branch2b-scale','res2b_branch2c-scale','res2b','res2c_branch2a-scale','res2c_branch2b-scale','res2c_branch2c-scale','res2c','res3a_branch2a-scale','res3a_branch2b-scale','res3a_branch2c-scale','res3a_branch1-scale','res3a','res3b_branch2a-scale','res3b_branch2b-scale','res3b_branch2c-scale','res3b','res3c_branch2a-scale','res3c_branch2b-scale','res3c_branch2c-scale','res3c','res3d_branch2a-scale','res3d_branch2b-scale','res3d_branch2c-scale','res3d','res4a_branch2a-scale','res4a_branch2b-scale','res4a_branch2c-scale','res4a_branch1-scale','res4a','res4b_branch2a-scale','res4b_branch2b-scale','res4b_branch2c-scale','res4b','res4c_branch2a-scale','res4c_branch2b-scale','res4c_branch2c-scale','res4c','res4d_branch2a-scale','res4d_branch2b-scale','res4d_branch2c-scale','res4d','res4e_branch2a-scale','res4e_branch2b-scale','res4e_branch2c-scale','res4e','res4f_branch2a-scale','res4f_branch2b-scale','res4f_branch2c-scale','res4f','res5a_branch2a-scale','res5a_branch2b-scale','res5a_branch2c-scale','res5a_branch1-scale','res5a','res5b_branch2a-scale','res5b_branch2b-scale','res5b_branch2c-scale','res5b','res5c_branch2a-scale','res5c_branch2b-scale','res5c_branch2c-scale','res5c','pool5','fc1000']

layer_count = 0
filter_count = 0
feature_file_count = 0
def FeatureWrite(name,x):
    with open('Feature_int8/resnet50/'+name+'.txt','w') as data:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        st = x[i][j][k][l].item()
                        st = str(st)
                        data.write(st)
                        data.write('\n')

def FeatureWriteFC(name,x):
    with open('Feature_int8/resnet50/'+name,'w') as data:
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                st = x[i][j].item()
                st = str(st)
                data.write(st)
                data.write('\n')    

def FeatureWriteFC_Bin(name,x):
    with open('Feature_int8/resnet50/'+name,'wb') as data:
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                st = x[i][j].item()
                st = struct.pack('f',st)
                data.write(st)


def GetQ(Q):    
    global layer_name_binQ
    q_list = []
    for m in range(51):
        Q_key = layer_name_binQ[m]
        with open('../channel_q/resnet50/' + Q_key,'r') as data:
            line = data.readline()
            linedata = line.strip().split(' ')
            for item in linedata:
                item = np.int8(item)
                q_list.append(item)
        print(len(q_list))
        Q[Q_key] = q_list
        q_list = []
    return(Q)        

def GetRealWeight(Filter,Q):
    global filter_name,layer_name_binQ,INFLAT
    filter_count = 0
    for i in range(50):
        print('**********')
        if i == 3 or i == 12 or i == 24 or i == 42:
            res = 2
        else:
            res = 1    
        Q1_key  = layer_name_binQ[i] 
        Q2_key = layer_name_binQ[i+1]
        for n in range(res):
            if n == 1:
                Q1_key = layer_name_binQ[i-2]
            filter_key = filter_name[filter_count]
            print(filter_key)
            out_c = Filter[filter_key].shape[0]
            in_c = Filter[filter_key].shape[1]
            for j in range(out_c):
                Filter[filter_key][j] = Filter[filter_key][j] + Q[Q2_key][j] + INFLAT
                for k in range(in_c):
                    Filter[filter_key][j][k] =  Filter[filter_key][j][k] - Q[Q1_key][k]
            filter_min = np.min(Filter[filter_key])
            filter_max = np.max(Filter[filter_key])
            print(filter_min)
            print(filter_max)
            Filter[filter_key][Filter[filter_key]<0] = 0
            filter_count = filter_count + 1
            print('**********')
    return (Filter)

def GetBias(bias,Q2_key):
    global layer_name_binQ,INFLAT,Q
    #Q2_key = layer_name_binQ[1]      
    out_c = len(Q[Q2_key])
    bias_power = np.zeros((out_c))
    bias_power = np.float32(bias_power)
    bias = np.array(bias)
    for i in range(out_c):
        power_num = Q[Q2_key][i] + INFLAT 
        bias_power[i] = bias[i]*pow(2.,power_num)
    return (bias_power)

def Conv2dInt8(input_data,weight,weight2,stride,padding):
    npd = ((0,0),(0,0),(padding,padding),(padding,padding))
    input_data = np.lib.pad(input_data,npd,'constant',constant_values=0)
    input_size = input_data.shape 
    weight_size = weight.shape
    N = input_size[0]
    in_c = input_size[1]
    out_c = weight_size[0]

    out_h = int((input_size[2] - weight_size[2])/stride + 1)
    out_w = int((input_size[3] - weight_size[3])/stride + 1)
    conv_result = np.ones((N,out_c,out_h,out_w))
    conv_result = np.int32(conv_result)
    inputdata = np.int8(input_data) 
    weight = np.int32(weight)
    weight2 = np.int8(weight2)
    for i in range(N):
        for j in range(out_c):
            for k in np.arange(input_size[2] - weight_size[2] + 1)[::stride]:
                for l in np.arange(input_size[3] - weight_size[3] + 1)[::stride]:
                    conv_result[i,j,k//stride,l//stride] = np.sum((input_data[i,:,k:k + weight_size[2],l:l + weight_size[3]]<<weight[j,:])*weight2[j,:])
    return (conv_result)

def BN(conv_result,bias_power,alpha,beta):
    global layer_count,layer_name_binQ,Q,INFLAT
    Q2_key = layer_name_binQ[layer_count] 
    N = conv_result.shape[0]
    out_c = conv_result.shape[1]
    out_h = conv_result.shape[2]
    out_w = conv_result.shape[3]
    bias_power = np.float32(bias_power)
    bn_result = np.zeros((N,out_c,out_h,out_w))
    bn_result = np.float32(bn_result)
    for i in range(N):
        for j in range(out_c):
            bn_result[i][j] = (alpha[j]*(conv_result[i][j] + bias_power[j]) + beta[j]*pow(2.,(Q[Q2_key][j]+INFLAT)))*pow(2.,-INFLAT)
    bn_result = np.round(bn_result)
    return (bn_result)

def FC(x,weight,weight2,bias):
    global BatchSize
    x = np.int8(x)
    weight = np.int32(weight)
    weight2 = np.int8(weight2)
    print("the shape of the output is :",end=' ')
    print(weight.shape[0])
    out = np.ones((BatchSize,weight.shape[0]))
    out = np.int32(out)
    for i in range(BatchSize):
        for j in range(weight.shape[0]):
            out[i,j] = np.sum((x[i,:]<<weight[j,:])*weight2[j,:])#bias[j]
    out = torch.Tensor(out)
    bias = torch.Tensor(bias)
    out[0,:] = out[0,:] + bias[:]
    out[0,:] = out[0,:]*pow(2.,-INFLAT)
    out = torch.round(out)
    out[out<-128] = -128
    out[out>127] = 127
    return (out)

"""
class PreProcess(object):
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
"""

BatchSize= 1
data_dir = '/data/yutong/imagenet'
#transform = transforms.Compose([PreProcess()])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
val = datasets.ImageFolder(os.path.join(data_dir,'val'),transform)
ValLoader = data.DataLoader(val,batch_size=BatchSize,shuffle=False)

# ResNet50

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        global layer_name_bin,filter_name,bn_name,layer_count,filter_count,feature_file_count,Filter,Filter2,alpha,beta
        stride = self.stride
        x = np.int8(x)
        identity = x
        filter_key = filter_name[filter_count]
        bn_key = bn_name[filter_count]
        out = Conv2dInt8(x,Filter[filter_key],Filter2[filter_key],1,0)
        layer_count = layer_count + 1
        feature_file_count = feature_file_count + 1
        bias = np.zeros((out.shape[1]))
        print('FilterCount = ',end=' ')
        print(filter_count)
        out = BN(out,bias,alpha[bn_key],beta[bn_key])
        out[out<-128] = -128
        out[out>127] = 127
        filter_count = filter_count + 1
        FeatureWrite(layer_name_bin[feature_file_count],out)
        out = torch.Tensor(out)
        out = self.relu(out)
 
        filter_key = filter_name[filter_count]
        bn_key = bn_name[filter_count]
        out = np.int8(out)
        out = Conv2dInt8(out,Filter[filter_key],Filter2[filter_key],stride,1) #padding=dialation
        layer_count = layer_count + 1
        feature_file_count = feature_file_count + 1
        bias = np.zeros((out.shape[1]))
        out = BN(out,bias,alpha[bn_key],beta[bn_key])
        out[out<-128] = -128
        out[out>127] = 127
        filter_count = filter_count + 1
        FeatureWrite(layer_name_bin[feature_file_count],out)
        out = torch.Tensor(out)
        out = self.relu(out)

        filter_key = filter_name[filter_count]
        print(len(bn_name))
        print(filter_count)
        bn_key = bn_name[filter_count]
        out = np.int8(out)
        out = Conv2dInt8(out,Filter[filter_key],Filter2[filter_key],1,0)
        layer_count = layer_count + 1
        feature_file_count = feature_file_count + 1
        bias = np.zeros((out.shape[1]))
        out = BN(out,bias,alpha[bn_key],beta[bn_key])
        filter_count = filter_count + 1
        #out = np.int16(out)
        out[out<-128] = -128
        out[out>127] = 127
        FeatureWrite(layer_name_bin[feature_file_count],out)
        out = np.int16(out)
        if self.downsample is not None:
            filter_key = filter_name[filter_count]
            bn_key = bn_name[filter_count]
            identity = Conv2dInt8(identity,Filter[filter_key],Filter2[filter_key],stride,0)
            feature_file_count = feature_file_count + 1 # add by shenfw
            bias = np.zeros((identity.shape[1]))
            identity = BN(identity,bias,alpha[bn_key],beta[bn_key])
            #identity = np.int16(identity)
            identity[identity<-128] = -128
            identity[identity>127] = 127
            FeatureWrite(layer_name_bin[feature_file_count],identity) # add by shenfw
            identity = np.int16(identity)
            filter_count = filter_count + 1
        
        out += identity
        out[out>127] = 127
        out[out<-128] = -128
        #FeatureWrite(layer_name_bin[layer_count],out)
        out = torch.Tensor(out)
        out = self.relu(out)   
        feature_file_count = feature_file_count + 1
        FeatureWrite(layer_name_bin[feature_file_count],out)
        return out

class ResNet(nn.Module):
    
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,256,layers[2],stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,512,layers[3],stride=2,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m,BasicBolck):
                    nn.init.constant_(m.bn2.weight,0)
                    
    def _make_layer(self,block,planes,blocks,stride=1,dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                norm_layer(planes*block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        global Q, fc_weight, fc_bias, layer_name_bin,filter_name,bn_name,layer_count,filter_count,feature_file_count,Filter,Filter2,bias_power,fc_bias,alpha,beta
        filter_key = filter_name[filter_count]
        bn_key = bn_name[filter_count]
        
        x = Conv2dInt8(x,Filter[filter_key],Filter2[filter_key],2,3)
        feature_file_count = feature_file_count + 1
        layer_count = layer_count + 1
        bias_power = np.zeros((x.shape[1]))
        x = BN(x,bias_power,alpha[bn_key],beta[bn_key])
        filter_count = filter_count + 1
        x[x<-128] = -128
        x[x>127] = 127
        FeatureWrite(layer_name_bin[feature_file_count],x)
        x = torch.Tensor(x)
        x = self.relu(x)
        #npd = ((0,0),(0,0),(0,1),(0,1))
        #x = np.lib.pad(x,npd,'constant',constant_values=0)
        #pd = (0,1,0,1)
        #x = F.pad(x,pd,'constant',0)
        x = self.maxpool(x)
        FeatureWrite('pool1',x);
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        x = self.avgpool(x)
        x = torch.round(x)
        FeatureWriteFC('pool5.txt',x)
        x = x.view(x.size(0),-1)
        Q2_key = layer_name_binQ[50]
        bias = GetBias(fc_bias,Q2_key)
        print(filter_count)
        print(len(filter_name))
        filter_key = filter_name[filter_count]
        out = FC(x,Filter[filter_key],Filter2[filter_key],bias)
        
        FeatureWriteFC('fc1000.txt',out)
        filter_count = 0
        layer_count = 0
        feature_file_count = 0
        print('end')
        Q_fc = Q[layer_name_binQ[50]]
        for i in range(1000):
            out[0,i] = out[0,i]*pow(2.,-Q_fc[i]) 
        FeatureWriteFC('fc-dequantization.txt',out)
        return out
    
    def resnet50(pretrained=False,**kwargs):
        model = ResNet(Bottleneck,[3,4,6,3],**kwargs)
        return model

cnn = ResNet.resnet50()
cnn.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = tvmodel.resnet50()#.to(device)
model2 = torch.nn.DataParallel(model2)#.cuda()
model2.eval()
checkpoint = torch.load('../weights/resnet50.pth.tar',map_location='cpu')
model2.load_state_dict(checkpoint['state_dict'])

filter_name = []
bn_name = []
INFLAT = np.int8(15) #15
Q = OrderedDict()
Filter = OrderedDict()
Filter2 = OrderedDict()
alpha = OrderedDict()
beta = OrderedDict()
bias = []
fc_weight = torch.ones(1000,2048)
fc_bias = []
Pars = model2.state_dict()
count = 0
for key in Pars:
    if count%6 == 0:
        Pars2  = torch.abs(Pars[key])
        Pars2[Pars2>0] = torch.round(torch.log2(Pars2[Pars2>0]))
        Filter[key] = Pars2
        Filter[key] = Filter[key].cpu().numpy()
        Filter[key] = np.int8(Filter[key])
        Filter2[key] = Pars[key]
        Filter2[key][abs(Filter2[key])<0.00001] = 0
        Filter2[key][Filter2[key]<0] = -1
        Filter2[key][Filter2[key]>0] = 1
        Filter2[key] = Filter2[key].cpu().numpy()
        Filter2[key] = np.int8(Filter2[key])
        filter_name.append(key)
        if count == 318:
            fc_weight = Pars[key].cpu().numpy()
    #if count == 1: // pytorch has no bias for conv
    #    bias_num = Pars[key].size(0)
    #    for i in range(bias_num):
    #        bias.append(Pars[key][i]) 
    if count == 319:
        bias_num = Pars[key].size(0)
        for i in range(bias_num):
            fc_bias.append(Pars[key][i])
    if (count-1)%6 == 0 and count != 319:
        alpha0 = Pars[key]
    if (count-2)%6 == 0:
        beta0 = Pars[key]
    if (count-3)%6 == 0:
        mean = Pars[key]
    if (count-4)%6 == 0:
        bn_name.append(key)
        var = Pars[key]
        alpha[key] = alpha0/(torch.sqrt(var+0.00001))
        beta[key] = -alpha[key]*mean + beta0 
        alpha[key] = alpha[key].cpu().numpy()
        beta[key] = beta[key].cpu().numpy()  
         
    count = count + 1

Q = GetQ(Q)
Filter = GetRealWeight(Filter,Q)

Q2_key = layer_name_binQ[1]
#bias_power = GetBias(bias,Q2_key)
#bias_power = np.array(bias_power)

def accuracy(output,target,topk=(1,5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    print(pred)
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

InputList = []
InputX = np.zeros((1,3,224,224))
print(InputX.shape)
with open('Feature_float/resnet50/data.bin','rb') as fig:
    for i in range(150528):
        data = fig.read(4)
        data_float = struct.unpack("f",data)[0]
        InputList.append(data_float)

Index = 0
for i in range(1):
    for j in range(3):
        for k in range(224):
            for l in range(224):
                InputX[i][j][k][l] = InputList[Index]
                Index = Index + 1
for i in range(InputX.shape[1]):
    InputX[0][i] = InputX[0][i]*pow(2,5)

InputX = np.round(InputX)
InputX[InputX<-128] = -128
InputX[InputX>127] = 127
InputX = np.int8(InputX)
Temp1 = []
Temp5 = []

for j,(test_x,test_y) in enumerate(ValLoader):
    test_x = InputX
    print('The test x is:')
    print(test_x)
    output_test = cnn(test_x)
    prec1,prec5 = accuracy(output_test.data,test_y,topk=(1,5))
    prec1 = prec1.cpu().numpy()
    prec5 = prec5.cpu().numpy()
    Temp1.append(prec1[0])
    Temp5.append(prec5[0])
    break

Top1 = np.array(Temp1)
Top5 = np.array(Temp5)
print('Top1 = ',end=' ')
print(Top1.mean(),end=' ')
print('Top5 = ',end=' ')
print(Top5.mean())
#"""
