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

import os
import torch
import torch.nn as nn
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torchvision.models as tvmodel

from ConvertModel import ConvertModel_caffe
""" Import your net structure here """

"""  Set empty path to use default weight initialization  """
# model_path = '../ModelFiles/ResNet/resnet50.pth'
model_path = '../finetune.pth'
ModelDir = '../ModelFiles/'

InputShape=[1,3,224,224]
"""  Init pytorch model  """
pytorch_net = tvmodel.resnet50()
NetName = str(pytorch_net.__class__.__name__)
if model_path != '':
    try:
        #pytorch_net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        finetune_checkpoint = torch.load(model_path)
        pytorch_net=finetune_checkpoint['model']
        pytorch_net = nn.DataParallel(pytorch_net)
    except AttributeError:
        pytorch_net = torch.load(model_path, map_location=lambda storage, loc: storage)
else:
    if not os.path.exists(ModelDir + NetName):
        os.makedirs(ModelDir + NetName)
    print('Saving default weight initialization...')
    torch.save(pytorch_net.state_dict(), ModelDir + NetName + '/' + NetName + '.pth')
"""  Connnnnnnnvert!  """
print('Converting...')
#pytorch_net = pytorch_net.to('cpu')
#for parameter in pytorch_net.parameters():
    #print(parameter)
text_net, binary_weights = ConvertModel_caffe(pytorch_net, InputShape, softmax=False)

"""  Save files  """
if not os.path.exists(ModelDir + NetName):
    os.makedirs(ModelDir + NetName)
print('Saving to ' + ModelDir + NetName)


import google.protobuf.text_format
with open(ModelDir + NetName + '/' + NetName + '.prototxt', 'w') as f:
    print('Saving to prototxt')
    f.write(google.protobuf.text_format.MessageToString(text_net))
with open(ModelDir + NetName + '/' + NetName + '.caffemodel', 'wb') as f:
    print('Saving to caffemodel')
    f.write(binary_weights.SerializeToString())

print('Converting Done done.')


