/* Copyright 2019 Inspur Corporation. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef __CYCLES_COMPUTATION_H__
#define __CYCLES_COMPUTATION_H__
#include "tf2_auto_param.h"
int tf2_auto_find_input_reader_cycles(int offset_subnet);
int tf2_auto_find_feature_writer_cycles(module_framework_data* module_framework, int c_index);
int tf2_auto_find_filter_reader_cnct_cycles(int c_index, int cnct);
int tf2_auto_find_filter_reader_conv_cycles(int c_index);
int tf2_auto_find_filter_reader_conv_total_cycles(int index_subnet, int offset_subnet);
int tf2_auto_find_filter_preload_cycles(int offset_subnet);
int tf2_auto_find_conv_cnct_cycles(int c_index, int cnct, int index_subnet);
int tf2_auto_find_conv_cycles(int c_index, int index_subnet);
//int tf2_auto_find_pool_cnct_cycles(int c_index, int cnct);
int tf2_auto_find_pool_conv_cycles(int c_index);
int tf2_auto_find_pool_total_cycles(int index_subnet, int offset_subnet);
int tf2_auto_find_conv_write_cache_cycles(module_framework_data* module_framework, int c_index);
int tf2_auto_find_conv_write_cache_total_cycles(module_framework_data* module_framework);
int tf2_auto_find_end_pool_total_cycles(module_framework_data* module_framework);
bool cycles_computation(module_framework_data* module_framework);
#endif