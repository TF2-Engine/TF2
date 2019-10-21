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
from torchsummary import *
def generate_prunedmodel(pytorch_net,conv_dnscp_flag_list,conv_mask_list):
	torch.save(pytorch_net,'model.pkl')
	pruned_model = torch.load('model.pkl')
	os.remove('model.pkl')
	convbn_block_mask_dict = splitnet_to_layerblock(pruned_model,conv_dnscp_flag_list,conv_mask_list)
	for blockindex,module_mask_tuple in convbn_block_mask_dict.items():
		if len(module_mask_tuple[1])>1:
			modelblob_convert(module_mask_tuple)
	return pruned_model
def calculate_compress_rate(src_model,pruned_model,inputshape):
	#pruned_model = generate_prunedmodel(pytorch_net,conv_dnscp_flag_list,conv_mask_list)
	tuple_inputshape=(inputshape[1],inputshape[2],inputshape[3])
	srcconv_total_params,srcconv_total_flops = summary(src_model, tuple_inputshape)
	prunedconv_total_params,prunedconv_total_flops = summary(pruned_model, tuple_inputshape)
	convparam_kept = prunedconv_total_params.item()/srcconv_total_params.item()
	convflops_kept = prunedconv_total_flops.item()/srcconv_total_flops.item()
	#print(srcconv_total_params,srcconv_total_flops)
	#print(prunedconv_total_params,prunedconv_total_flops)
	return convparam_kept,convflops_kept
def clip_cin_channel(tensor_weightdata,tensor_maskdata):
	count = tensor_maskdata.numel()
	c_num = tensor_maskdata.size()[1]
	count_nhw = count/c_num
	sumMask_C   = torch.abs(tensor_maskdata).sum(3).sum(2).sum(0)/count_nhw
	w_kept_cin_index = torch.nonzero(sumMask_C).reshape(-1)
	#print(w_kept_cin_index.size())
	weight_kept = tensor_weightdata[:,w_kept_cin_index,:,:]
	return weight_kept,w_kept_cin_index
def clip_cout_channel(tensor_weightdata,w_kept_cin_index):
	if len(tensor_weightdata.shape)==4 or len(tensor_weightdata.shape)==1:
		if len(tensor_weightdata.shape)==4:
			weight_kept = tensor_weightdata[w_kept_cin_index,:,:,:]
		if len(tensor_weightdata.shape)==1:
			weight_kept = tensor_weightdata[w_kept_cin_index]
		return weight_kept
	else:
		return tensor_weightdata
def modelblob_convert(module_mask_tuple):
	modules = module_mask_tuple[0]
	conv_mask_list = module_mask_tuple[1]
	wkept_cin_list = []
	conv_index = 0
	'''clip cin'''
	for moduleindex,module in enumerate(modules):
		if isinstance(module,nn.Conv2d):
			if conv_index>0:
				tensor_weightdata = module.weight.data
				tensor_maskdata = conv_mask_list[conv_index]
				weight_kept,w_kept_cin_index = clip_cin_channel(tensor_weightdata,tensor_maskdata)
				module.weight.data = weight_kept
				module.in_channels = weight_kept.shape[1]
				wkept_cin_list.append(w_kept_cin_index)
			conv_index += 1
	'''clip cout'''
	conv_index = 0
	for moduleindex,module in enumerate(modules):
		if isinstance(module,nn.Conv2d):
			conv_index += 1
		if conv_index!= len(conv_mask_list):
			if isinstance(module,nn.Conv2d):
				tensor_weightdata = module.weight.data
				w_kept_cin_index = wkept_cin_list[conv_index-1]
				weight_kept = clip_cout_channel(tensor_weightdata,w_kept_cin_index)
				module.weight.data = weight_kept
				module.out_channels = weight_kept.shape[0]
			elif isinstance(module,nn.BatchNorm2d):
				w_kept_cin_index = wkept_cin_list[conv_index-1]
				if module.affine:
					tensor_weightdata = module.weight.data
					weight_kept = clip_cout_channel(tensor_weightdata,w_kept_cin_index)
					module.weight.data = weight_kept
					tensor_weightdata = module.bias.data
					weight_kept = clip_cout_channel(tensor_weightdata,w_kept_cin_index)
					module.bias.data = weight_kept
				if module.track_running_stats:
					tensor_weightdata = module.running_mean
					weight_kept = clip_cout_channel(tensor_weightdata,w_kept_cin_index)
					module.running_mean.data = weight_kept
					tensor_weightdata = module.running_var
					weight_kept = clip_cout_channel(tensor_weightdata,w_kept_cin_index)
					module.running_var.data = weight_kept
				module.num_features = weight_kept.shape[0]
				

def splitnet_to_layerblock(pytorch_net,conv_dnscp_flag_list,conv_mask_list):
	'''every block one input and one output,that means one line'''
	'''0:conv no cp 1:conv cp -1:bn and other layer'''
	convbn_module_list = []
	convbn_flag_list =[]
	convblock_list = []
	conv_index=0
	for module in pytorch_net.modules():
		if isinstance(module,nn.Conv2d):
			convbn_module_list.append(module)
			convbn_flag_list.append(conv_dnscp_flag_list[conv_index])
			conv_index += 1
		if isinstance(module,nn.BatchNorm2d):
			convbn_module_list.append(module)
			convbn_flag_list.append(-1)
	if conv_index!=len(conv_dnscp_flag_list):
		print("conv num must same",conv_index,len(conv_dnscp_flag_list))
	convbn_block_dict = {}
	splitindex = 0
	for index,convbnflag in enumerate(convbn_flag_list):
		if index == 0:
			splitindex = index
			convblock_list.append(convbn_module_list[0])
		if convbnflag != 0:
			convblock_list.append(convbn_module_list[index])
		if convbnflag == 0 and index > 0:
			convbn_block_dict[splitindex] = (convblock_list)
			convblock_list = []
			splitindex = index
			convblock_list.append(convbn_module_list[index])
	convbn_block_dict[splitindex] = (convblock_list)

	'''descreption block info:index modules convmask'''
	total_convnum = 0
	convbn_block_mask_dict = {}
	for blockindex, modules in convbn_block_dict.items():
		convblock_mask_list = []
		for module in modules:
			if isinstance(module,nn.Conv2d):
				convblock_mask_list.append(conv_mask_list[total_convnum])
				total_convnum += 1
		convbn_block_mask_dict[blockindex] = (modules,convblock_mask_list)

	'''for blockindex,module_mask_tuple in convbn_block_mask_dict.items():
		print(blockindex,len(module_mask_tuple[1]))'''

	return convbn_block_mask_dict

