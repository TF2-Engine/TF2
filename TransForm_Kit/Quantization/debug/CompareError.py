import struct
import torch
import numpy as np
from collections import OrderedDict

LayerNameBinQ = ['conv1-scale','conv1-scale','res2a_branch2a-scale','res2a_branch2b-scale','res2c','res2c','res2c','res2b_branch2a-scale','res2b_branch2b-scale','res2c','res2c','res2c_branch2a-scale','res2c_branch2b-scale','res2c','res2c','res3a_branch2a-scale','res3a_branch2b-scale','res3d','res3d','res3d','res3b_branch2a-scale','res3b_branch2b-scale','res3d','res3d','res3c_branch2a-scale','res3c_branch2b-scale','res3d','res3d','res3d_branch2a-scale','res3d_branch2b-scale','res3d','res3d','res4a_branch2a-scale','res4a_branch2b-scale','res4f','res4f','res4f','res4b_branch2a-scale','res4b_branch2b-scale','res4f','res4f','res4c_branch2a-scale','res4c_branch2b-scale','res4f','res4f','res4d_branch2a-scale','res4d_branch2b-scale','res4f','res4f','res4e_branch2a-scale','res4e_branch2b-scale','res4f','res4f','res4f_branch2a-scale','res4f_branch2b-scale','res4f','res4f','res5a_branch2a-scale','res5a_branch2b-scale','res5c','res5c','res5c','res5b_branch2a-scale','res5b_branch2b-scale','res5c','res5c','res5c_branch2a-scale','res5c_branch2b-scale','res5c','res5c','res5c','fc1000']

LayerNameBin = ['conv1-scale','pool1','res2a_branch2a-scale','res2a_branch2b-scale','res2a_branch2c-scale','res2a_branch1-scale','res2a','res2b_branch2a-scale','res2b_branch2b-scale','res2b_branch2c-scale','res2b','res2c_branch2a-scale','res2c_branch2b-scale','res2c_branch2c-scale','res2c','res3a_branch2a-scale','res3a_branch2b-scale','res3a_branch2c-scale','res3a_branch1-scale','res3a','res3b_branch2a-scale','res3b_branch2b-scale','res3b_branch2c-scale','res3b','res3c_branch2a-scale','res3c_branch2b-scale','res3c_branch2c-scale','res3c','res3d_branch2a-scale','res3d_branch2b-scale','res3d_branch2c-scale','res3d','res4a_branch2a-scale','res4a_branch2b-scale','res4a_branch2c-scale','res4a_branch1-scale','res4a','res4b_branch2a-scale','res4b_branch2b-scale','res4b_branch2c-scale','res4b','res4c_branch2a-scale','res4c_branch2b-scale','res4c_branch2c-scale','res4c','res4d_branch2a-scale','res4d_branch2b-scale','res4d_branch2c-scale','res4d','res4e_branch2a-scale','res4e_branch2b-scale','res4e_branch2c-scale','res4e','res4f_branch2a-scale','res4f_branch2b-scale','res4f_branch2c-scale','res4f','res5a_branch2a-scale','res5a_branch2b-scale','res5a_branch2c-scale','res5a_branch1-scale','res5a','res5b_branch2a-scale','res5b_branch2b-scale','res5b_branch2c-scale','res5b','res5c_branch2a-scale','res5c_branch2b-scale','res5c_branch2c-scale','res5c','pool5','fc1000']

C = 64
H = 56
W = 56
FileCount = 0
Q = OrderedDict()
def QuantizeChannel(x,Q):
    for i in range(x.size(1)):
        x[:,i] = x[:,i]*pow(2,Q[i])
    return (x)

def GetQ(Q):
    global LayerNameBinQ
    q_list = []
    for m in range(72):
        Q_key = LayerNameBinQ[m]
        with open('../channel_q/resnet50/' + Q_key,'r') as data:
            line = data.readline()
            linedata = line.strip().split(' ')
            for item in linedata:
                item = np.int8(item)
                q_list.append(item)
        Q[Q_key] = torch.Tensor(q_list)
        q_list = []
    return (Q)
Q = GetQ(Q)
for m in range(72):# this 'if' block could be simpled
    FileCount = m
    if m == 0:
        C = 64
        H = 112
        W = H
    if m>0 and m<15:
        C = 64
        H = 56
        W = H
        if m == 4 or m == 5 or m == 6 or m == 9 or m == 10 or m == 13 or m == 14:
            C = C*4
    if m>14 and i<32:
        C = 128
        H = 28
        W = H
        if m == 17 or m == 18 or m == 19 or m == 22 or m == 23 or m == 26 or m == 27 or m == 30 or m == 31:
            C = C*4
    if m>31 and i<57:
        C = 256
        H = 14
        W = H
        if m == 34 or m == 35 or m ==36 or m == 39 or m == 40 or m == 43 or m == 44 or m == 47 or m == 48 or m == 51 or m == 52 or m == 55 or m == 56:
            C = C*4
    if m>56 and i<70:
        C = 512
        H = 7
        W = H
        if m == 59 or m == 60 or m == 61 or m == 64 or m == 65 or m == 68 or m == 69:
            C = C*4
    if m>69:
        H = 1
        W = H
        if m == 70:
            C = 2048
        else:
            C = 1000
    if m < 72:
        with open('Feature_float/resnet50/'+LayerNameBin[m]+'.bin',"rb") as conv1:
            Fea1 = np.zeros((1,C,H,W))
            for i in range(1):
                for j in range(C):
                    for k in range(H):
                        for l in range(W):
                            data = conv1.read(4)
                            data_float = struct.unpack("f",data)[0]
                            Fea1[i][j][k][l] = data_float
    Fea1 = torch.Tensor(Fea1)
    Q_key = LayerNameBinQ[m]
    Fea1 = QuantizeChannel(Fea1,Q[Q_key])
    with open('Feature_int8/resnet50/'+LayerNameBin[m]+'.txt',"r") as Qdata:
        QFea = np.zeros((1,C,H,W))
        for i in range(1):
            for j in range(C):
                for k in range(H):
                    for l in range(W):
                        data = float(Qdata.readline())
                        QFea[i][j][k][l] = data
        QFea = torch.Tensor(QFea)
    TotalFea1 = 0
    TotalError = 0
    MaxError = 0
    MaxFea1 = 0

    Fea1 = np.array(Fea1)
    QFea = np.array(QFea)
    QError = abs(QFea-Fea1)
    Fea1 = abs(Fea1)

    MaxError = np.max(QError)
    MaxFea1 = np.max(Fea1)

    TotalError = QError.sum()
    TotalFea1 = Fea1.sum()

    print(m,end=' ')
    print(LayerNameBin[m],end=' ')
    print(100*TotalError/TotalFea1,"%")

