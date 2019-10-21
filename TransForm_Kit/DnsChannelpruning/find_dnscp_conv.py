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
import torch.nn as nn

def find_dnscp_conv(pytorch_net,layername_type_dict,layername_connect_dict):
	'''key:convlayername value:cpflag 1:cp 0:no cp'''
	conv_dnscp_dict = {} 

	'''find all convlayer use layertype(backward funcname) default:all conv layer need cp'''
	for layername,layertype in layername_type_dict.items():
		if layertype.find('Conv2D')>0 or layertype.find('CudnnConvolution')>=0:
			conv_dnscp_dict[layername] = 1
			#print("backward conv info:",layername,conv_dnscp_dict[layername])
	'''find convlayer that cannot be cp,for example this conv and another op has the same input'''
	for layername,layername_connect_list in layername_connect_dict.items():
		if len(layername_connect_list)>1 and layername_type_dict[layername] != "Broadcast":
			for layername_connect in layername_connect_list:
				if layername_connect in conv_dnscp_dict:
					conv_dnscp_dict[layername_connect] = 0
	'''temp print'''
	for k,v in conv_dnscp_dict.items():
		print(k,v)
	'''find all convlayer use forwad''' 
	conv_dnscp_flag_list=[]
	for layer in pytorch_net.modules():
		if isinstance(layer,nn.Conv2d):
			conv_dnscp_flag_list.append(1)

	'''we think backward(reserve) and forward convolution are in the same order'''
	if len(conv_dnscp_flag_list)!=len(conv_dnscp_dict):
		print("forward conv num:",len(conv_dnscp_flag_list),"backward conv num:",len(conv_dnscp_dict))
		print("forward and backward conv num must be same") 
	else:
		conv_index = 0
		for convname,dnscp_flag in conv_dnscp_dict.items():
			conv_dnscp_flag_list[conv_index] = dnscp_flag
			conv_index += 1
		'''first conv cp flag must 0'''
		conv_dnscp_flag_list[0] = 0  
	return conv_dnscp_flag_list