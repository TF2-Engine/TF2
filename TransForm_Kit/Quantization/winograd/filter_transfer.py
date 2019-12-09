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

""" 1D Winograd, the filter should be transfered by the winograd filter transfer matrix. 1D winograd math:: A^T[(Gg)*(B^Td)], the filter transfer is 'Gg', where 'g' is the filter and the G is the transfer matrix, the '^T' means the transpose of the matrix. The 1D filter transfer implementation includes F(2,3), F(4,3), F(5,3), for the F(a,b), where the first number 'a' called 'trans_size'.

  2D Winograd, the filter should be transfered by the winograd filter transfer matrix. 2D winograd math:: A^T[(GgG^T)*(B^TdB)]A, the filter transfer is 'GgG^T', where 'g' is the filter and the G and G^T is the filter transfer matrix and its transpose, respectively.The 2D filter transfer implementation includes F(2x2,3x3), F(4x4,3x3), F(5x5,3x3), for the F(axa,bxb), where the first number 'a' called 'trans_size'. """

def FilterTransfer(filter_shape,filter_raw, filter_trans, trans_size, trans_dim):
    
    k = 0
    w = 3
    """ F(2,3) or F(2x2,3x3) """
    if trans_size = 2:
        G_list = [1,0,0,0.5,0.5,0.5,0.5,-0.5,0.5,0,0,1]
        h = 4
    """ F(4,3) or F(4x4,3x3) """
    if trans_size = 4:
        G_list = [6,0,0,-4,-4,-4,-4,4,-4,1,2,4,1,-2,4,0,0,24]
        h = 6
    """ F(5,3) or F(5x5,3x3) """
    if trans_size = 5: 
        G_list = [10,0,0,10,10,10,5,-5,5,-5,-10,-20,-1,2,-4,1,3,9,0,0,120]
        h = 7

    G = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            G[i][j] = G_list[k]
            k = k + 1 
    G = mat(G)

    """1D transfer"""
    if trans_dim = 1:
        for i in range(filter_shape[0]):
            for j in range(filter_shape[1]):
                filter_raw[i,j] = mat(filter_raw[i,j])
                filter_mid = G*(filter_raw[i,j].T)
                filter_trans[i,j] = filter_mid.T
    """2D transfer"""
    if trans_dim = 2:
        for i in range(filter_shape[0]):
            for j in range(filter_shape[1]):
                filter_raw[i,j] = mat(filter_raw[i,j])
                filter_mid = G*(filter_raw[i,j].T)
                filter_trans[i,j] = filter_mid*(G.T)
    
    return filter_trans
