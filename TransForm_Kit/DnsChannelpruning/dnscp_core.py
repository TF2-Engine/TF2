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
import random
import math

class DnscpCore():
	def __init__(self):
		self._gamma = 0.0000025
		self._power = 1
		self._c_rate = 0.1
		self._alpha_low = 0.95
		self._alpha_high = 1.05
		self._iter_stop = 100000
		self._param_mask_list = []
		self._convparam_cur_list = []
		self._std_list = []
		self._mu_list = []
	def maskChannelCalc(self,param,conv_index):
		param_mask = self._param_mask_list[conv_index]
		mu = self._mu_list[conv_index]
		std = self._std_list[conv_index]
		'''if conv_index ==6:
			print("param mask:",torch.abs(param_mask).sum(3).sum(2).sum(0),param_mask.size())'''
		count = param_mask.numel()
		c_num = param_mask.size()[1]
		count_nhw = count/c_num
		if len(param.size())!=4:
			print('param size must 4 dim')
		else:
			sumWeight_C = torch.abs(param).sum(3).sum(2).sum(0)/count_nhw
			sumMask_C   = torch.abs(param_mask).sum(3).sum(2).sum(0)/count_nhw
			for c_index in range(sumWeight_C.size()[0]):
				'''if conv_index == 6:
					print("conv_index:",conv_index,"channle:",c_index,sumMask_C[c_index], \
					  sumWeight_C[c_index],self._alpha_low*max(mu+self._c_rate*std,0),"c*h*w:",count_nhw)'''
				if sumMask_C[c_index]== 1 and sumWeight_C[c_index]<=self._alpha_low*max(mu+self._c_rate*std,0):
					param_mask[:,c_index,:,:]=0
				elif sumMask_C[c_index]== 0 and sumWeight_C[c_index]>self._alpha_high*max(mu+self._c_rate*std,0):
					param_mask[:,c_index,:,:]=1
		sumMask_C   = torch.abs(param_mask).sum(3).sum(2).sum(0)/count_nhw
		nz_c = torch.nonzero(sumMask_C).size()[0]
		final_c16 = round(float(nz_c)/16)*16 if round(float(nz_c)/16)*16>0 else 16
		if final_c16 > nz_c: #change 0 to 1
			ichange_c = final_c16 - nz_c
			for c_index in range(c_num):
				if sumMask_C[c_index]==0 and ichange_c>0:
					param_mask[:,c_index,:,:]=1
					ichange_c=ichange_c-1
		if final_c16 < nz_c: #change 1 to 0
			ichange_c = nz_c - final_c16
			for c_index in range(c_num):
				if sumMask_C[c_index]== 1 and ichange_c>0:
					param_mask[:,c_index,:,:]=0
					ichange_c=ichange_c-1
		'''print("convindex:",conv_index,final_c16,nz_c)'''
		self._param_mask_list[conv_index] = param_mask
	def cal_mean_std(self,iter,conv_index,param):
		if self._std_list[conv_index] == 0 and iter ==0:
			#weight sum and square
			ncount = self._param_mask_list[conv_index].numel()
			mtw = param*self._param_mask_list[conv_index]
			sum_mu = torch.sum(torch.abs(mtw))
			sum_square = torch.sum(mtw *param)
			nz_w = torch.nonzero(mtw).size()[0]
			self._mu_list[conv_index] = sum_mu/nz_w
			self._std_list[conv_index] = torch.sqrt((sum_square - \
		                   ncount*self._mu_list[conv_index]*self._mu_list[conv_index])/ncount)
			#print("conv_index",conv_index,'count:',ncount,'mu:',self._mu_list[conv_index],'std:',self._std_list[conv_index])

	def forward_dnscp_layer(self,pytorch_net,conv_dnscp_flag,iter):
		"""init conv mask"""
		conv_index = 0
		if iter == 0: 
			for layer in pytorch_net.modules():
				if isinstance(layer,nn.Conv2d):
					param = layer.weight.data
					param_mask = torch.ones_like(param)
					conv_index += 1
					self._std_list.append(0)
					self._mu_list.append(0)
					self._param_mask_list.append(param_mask)
					self._convparam_cur_list.append(param)
		"""cal conv mask and cal pruned param"""
		conv_index = 0 
		for layer in pytorch_net.modules():
			if isinstance(layer,nn.Conv2d):
				param = layer.weight.data
				self._convparam_cur_list[conv_index] = param
				"""if conv need to be pruned,cal pruned channel"""
				if conv_dnscp_flag[conv_index] == 1:
					if iter == 0:
						self.cal_mean_std(iter,conv_index,param)
					rnd = torch.rand(1).item()  
					"""TODO print param mask"""
					#print("cur mask:",self._param_mask_list[conv_index]) 
					#print("rnd:",rnd,math.pow(1+self._gamma*iter,-self._power))
					if math.pow(1+self._gamma*iter,-self._power)>rnd and iter < self._iter_stop:
						self.maskChannelCalc(param,conv_index)
						#print("dnscal mask:",self._param_mask_list[conv_index])
				"""update forward need conv param"""
				layer.weight.data = layer.weight.data*self._param_mask_list[conv_index]
				conv_index += 1
		return self._convparam_cur_list,self._param_mask_list