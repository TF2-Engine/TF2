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

def Conv1FeatureTransfer(feature):
    # the x is the feature after padding=3, 230x230 
    # the 4x4, 4x3, 3x4, 3x3 is the shape of the corresponding fitler
    npd = ((0,0),(0,0),(3,3),(3,3))
    feature = np.lib.pad(feature,npd,'constant',constant_values=0)
    feature_list = []
    feature_4x4 = np.zeros((feature.shape[0],feature.shape[1],115,115))
    feature_4x3 = np.zeros((feature.shape[0],feature.shape[1],115,115))
    feature_3x4 = np.zeros((feature.shape[0],feature.shape[1],115,115))
    feature_3x3 = np.zeros((feature.shape[0],feature.shape[1],115,115))
    dilation_h = 0
    dilation_w = 0
    for i in range(115):
        for j in range(115):
            feature_4x4[:,:,i,j] = feature[:,:,i+dilation_h,j+dilation_w]
            feature_4x3[:,:,i,j] = feature[:,:,i+dilation_h,j+1+dilation_w]
            feature_3x4[:,:,i,j] = feature[:,:,i+1+dilation_h,j+dilation_w]
            feature_3x3[:,:,i,j] = feature[:,:,i+1+dilation_h,j+1+dilation_w]
            dilation_w = dilation_w + 1
        dilation_w = 0
        dilation_h = dilation_h + 1
    mid_feature1 = np.zeros((feature.shape[0],feature.shape[1],114,114))
    mid_feature2 = np.zeros((feature.shape[0],feature.shape[1],114,114))
    mid_feature3 = np.zeros((feature.shape[0],feature.shape[1],114,114))
    mid_feature4 = np.zeros((feature.shape[0],feature.shape[1],114,114))
    mid_feature5 = np.zeros((feature.shape[0],feature.shape[1],114,114))#114,115
    mid_feature6 = np.zeros((feature.shape[0],feature.shape[1],114,114))#114,115
    mid_feature7 = np.zeros((feature.shape[0],feature.shape[1],114,114))#115,114
    mid_feature8 = np.zeros((feature.shape[0],feature.shape[1],114,114))#115,114
    mid_feature9 = np.zeros((feature.shape[0],feature.shape[1],114,114))#115,115
    mid_feature1 = feature_4x4[:,:,0:114,0:114]
    mid_feature2 = feature_4x4[:,:,0:114,1:115]
    mid_feature3 = feature_4x4[:,:,1:115,0:114]
    mid_feature4 = feature_4x4[:,:,1:115,1:115]
    mid_feature5 = feature_4x3[:,:,0:114,0:114]
    mid_feature6 = feature_4x3[:,:,1:115,0:114]
    mid_feature7 = feature_3x4[:,:,0:114,0:114]
    mid_feature8 = feature_3x4[:,:,0:114,1:115]
    mid_feature9 = feature_3x3[:,:,0:114,0:114]
    feature_list.append(mid_feature1)
    feature_list.append(mid_feature2)
    feature_list.append(mid_feature3)
    feature_list.append(mid_feature4)
    feature_list.append(mid_feature5)
    feature_list.append(mid_feature6)
    feature_list.append(mid_feature7)
    feature_list.append(mid_feature8)
    feature_list.append(mid_feature9)
    return feature_list

def Conv1FilterTransfer(weight):
    weight_list = []
    weight_4x4 = np.zeros((weight.shape[0],weight.shape[1],4,4))
    weight_4x3 = np.zeros((weight.shape[0],weight.shape[1],4,3))
    weight_3x4 = np.zeros((weight.shape[0],weight.shape[1],3,4))
    weight_3x3 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    dilation_h = 0
    dilation_w = 0
    #---------------No.1--------------
    for j in range(4):
        for k in range(4):
            weight_4x4[:,:,j,k] = weight[:,:,j+dilation_h,k+dilation_w]
            dilation_w = dilation_w + 1
        dilation_w = 0
        dilation_h = dilation_h + 1
    mid_weight1 = weight_4x4[:,:,0:3,0:3]
    weight_list.append(mid_weight1)
    mid_weight2 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight2[:,:,:,2] = weight_4x4[:,:,0:3,3]
    weight_list.append(mid_weight2)
    mid_weight3 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight3[:,:,2,:] = weight_4x4[:,:,3,0:3]
    weight_list.append(mid_weight3)
    mid_weight4 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight4[:,:,2,2] = weight_4x4[:,:,3,3]
    weight_list.append(mid_weight4)
    #--------------No.2---------------
    dilation_h = 0
    dilation_w = 0
    for j in range(4):
        for k in range(3):
            weight_4x3[:,:,j,k] = weight[:,:,j+dilation_h,k+1+dilation_w]
            dilation_w = dilation_w + 1
        dilation_w = 0
        dilation_h = dilation_h + 1
    mid_weight5 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight5 = weight_4x3[:,:,0:3,:]
    weight_list.append(mid_weight5)
    mid_weight6 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight6[:,:,2,:] = weight_4x3[:,:,3,:]
    weight_list.append(mid_weight6)
    #--------------No.3---------------
    dilation_h = 0
    dilation_w = 0
    for j in range(3):
        for k in range(4):
            weight_3x4[:,:,j,k] = weight[:,:,j+1+dilation_h,k+dilation_w]
            dilation_w = dilation_w + 1
        dilation_w = 0
        dilation_h = dilation_h + 1
    mid_weight7 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight7 = weight_3x4[:,:,:,0:3]
    weight_list.append(mid_weight7)
    mid_weight8 = np.zeros((weight.shape[0],weight.shape[1],3,3))
    mid_weight8[:,:,:,2] = weight_3x4[:,:,:,3]
    weight_list.append(mid_weight8)
    #--------------No.4---------------
    dilation_h = 0
    dilation_w = 0
    for j in range(3):
        for k in range(3):
            weight_3x3[:,:,j,k] = weight[:,:,j+1+dilation_h,k+1+dilation_w]
            dilation_w = dilation_w + 1
        dilation_w = 0
        dilation_h = dilation_h + 1
    weight_list.append(weight_3x3)

    return weight_list

def Conv1_7x7_2_3x3_transfer(feature_raw, filter_raw):
    feature_list = Conv1FeatureTransfer(feature_raw)
    weight_list = Conv1FilterTransfer(filter_raw)

