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
from numpy import mat

""" 1D Winograd, the input or the feature from bottom layer should be transfered by the winograd feature transfer matrix. 1D winograd math:: A^T[(Gg)*(B^Td)], the feature transfer is 'B^Td', where 'd' is the feature and the B is the transfer matrix, the '^T' means the transpose of the matrix. The 1D feature transfer implementation includes F(2,3), F(4,3), F(5,3), for the F(a,b), where the first number 'a' called 'trans_size'.

  2D Winograd, the feature should be transfered by the winograd feature transfer matrix. 2D winograd math:: A^T[(GgG^T)*(B^TdB)]A, the feature transfer is 'B^dB', where 'd' is the feature and the B and B^T is the feature transfer matrix and its transpose, respectively.The 2D feature transfer implementation includes F(2x2,3x3), F(4x4,3x3), F(5x5,3x3), for the F(axa,bxb), where the first number 'a' called 'trans_size'. """

def FeatureTransfer(feature_shape,feature_raw, feature_trans, trans_size, trans_dim):
    
    k = 0
    """ F(2,3) or F(2x2,3x3) """
    if trans_size = 2:
        BT_list = [1,0,-1,0,0,1,1,0,0,-1,1,0,0,-1,0,1]
        h,w = 4
    """ F(4,3) or F(4x4,3x3) """
    if trans_size = 4:
        BT_list = [4,0,-5,0,1,0,0,-4,-4,1,1,0,0,4,-4,-1,1,0,0,-2,-1,2,1,0,0,2,-1,-2,1,0,0,4,0,-5,0,1]
        h,w = 6
    """ F(5,3) or F(5x5,3x3) """
    if trans_size = 5:
        BT_list = [12,-4,-15,5,3,-1,0,0,12,8,-7,-2,1,0,0,-12,16,-1,-4,1,0,0,6,1,-7,-1,1,0,0,-6,5,5,-5,1,0,0,4,0,-5,0,1,0,0,-12,4,15,-5,-3,1]
        h,w = 7

    BT = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            BT[i][j] = BT_list[k]
            k = k + 1 
    BT = mat(BT)
    feature_trans = feature_raw.copy()

    """1D transfer"""
    if trans_dim = 1:
        for i in range(feature_shape[0]):
            for j in range(feature_shape[1]):
                feature_raw[i,j] = mat(feature_raw[i,j])
                feature_mid = BT*(feature_raw[i,j].T)
                feature_trans[i,j] = feature_mid.T
    """2D transfer"""
    for i in range(feature_shape[0]):
        for j in range(feature_shape[1]):
            feature_raw[i,j] = mat(feature_raw[i,j])
            feature_mid = BT*(feature_raw[i,j].T)
            feature_mid_t = feature_mid*(BT.T)
            feature_trans[i,j] = feature_mid_t
 
    return filter_trans
