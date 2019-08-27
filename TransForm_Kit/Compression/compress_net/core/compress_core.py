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
import sys
import math
def ComputeQuantumRange(filter_data,mask_data,num_quantum_values):
	python_min_int = -sys.maxsize-1
	max_value_tobe_quantized = python_min_int
	max_value_quantized = python_min_int
	quantum_values = np.zeros(2*num_quantum_values+1)
	updated = 0
	filter_data = filter_data.reshape(-1)
	mask_data = mask_data.reshape(-1)
	for k in range(filter_data.size):
		if mask_data[k]==1:
			if abs(filter_data[k])>max_value_tobe_quantized:
				max_value_tobe_quantized = abs(filter_data[k])
		elif mask_data[k]==0:
			if abs(filter_data[k])>max_value_quantized:
				max_value_quantized = abs(filter_data[k])
				updated+=1
		else:
			print("Mask value is not 0, nor 1, in tp_inner_product_layer")
	if updated == 0:
		max_quantum_exp_ = math.floor(math.log(4.0 * max_value_tobe_quantized / 3.0) / math.log(2.0))
	elif updated > 0 and updated <filter_data.size:
		max_quantum_exp_ = round(math.log(max_value_quantized) / math.log(2.0))
		max_tobe_quantized_exp_ = math.floor(math.log(4.0 * max_value_tobe_quantized / 3.0) / math.log(2.0))
		if max_quantum_exp_<max_tobe_quantized_exp_:
			print('tobe_quan > has_quan')
			sys.exit(-1)
	else:
		max_quantum_exp_ = -100
	min_quantum_exp_ = max_quantum_exp_ - num_quantum_values + 1	
	for k in range(num_quantum_values):
		quantum_values[k]=pow(2.0,max_quantum_exp_-k)
		quantum_values[2*num_quantum_values-k]=-quantum_values[k]
	return max_quantum_exp_,min_quantum_exp_

def ShapeIntoTwoPower(input_blob,mask_blob,previous_portion,current_portion,max_quantum_exp_,min_quantum_exp_):	
	if current_portion==0:
		return input_blob,mask_blob
	if max_quantum_exp_ == -100:
		return input_blob,mask_blob
	src_shape = input_blob.shape
	param = input_blob.reshape(-1)
	mask = mask_blob.reshape(-1)
	count = param.size
	num_not_yet_quantized = 0
	sorted_param=[]
	for i in range(count):
		if mask[i]==1:
			num_not_yet_quantized+=1
			sorted_param.append(abs(param[i]))
	num_init_not_quantized = round(float(num_not_yet_quantized) / (1.0 - previous_portion));
	num_not_tobe_quantized = round(float(num_init_not_quantized) * (1.0 - current_portion));
	num_tobe_update = num_not_yet_quantized - num_not_tobe_quantized
	print("portions: ",previous_portion * 100 ,"% -> ",current_portion * 100 ,"% ("
            ,"total: " ,float(count-num_not_yet_quantized)/count*100 ,"% -> ",float(count-num_not_tobe_quantized)/count*100,"%",")")
	print("init_not_quantized/total: "
            ,num_init_not_quantized ,"/" 
            ,count)
	print( "to_update/not_tobe_quantized/not_yet_quantized: " 
            ,num_tobe_update ,"/"
            ,num_not_tobe_quantized ,"/"
            ,num_not_yet_quantized )
	if num_tobe_update>0:
		sorted_param = np.array(sorted_param)
		sorted_param = np.sort(sorted_param)
		threshold_ = sorted_param[num_not_tobe_quantized]
		#print(threshold_,type(threshold_),type(param),param[0])
		for i in range(count):
			if mask[i] == 1:
				if param[i] >= threshold_:
					exp_ = math.floor(math.log(4.0*param[i]/3.0)/math.log(2.0))
					if exp_>=min_quantum_exp_:
						param[i] = pow(2.0,exp_)
					else:
						param[i] = 0.0
					mask[i] = 0
				elif param[i] <= -threshold_:
					exp_ = math.floor(math.log(-1*4.0*param[i]/3.0)/math.log(2.0))
					if exp_>=min_quantum_exp_:
						param[i] = -pow(2.0,exp_)
					else:
						param[i] = 0.0
					mask[i] = 0
	input_blob = param.reshape(src_shape)
	mask_blob = mask.reshape(src_shape)
	return input_blob,mask_blob
		
		
