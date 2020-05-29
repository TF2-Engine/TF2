  
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
#ifndef __TF2_AUTO_PARAM_H__
#define __TF2_AUTO_PARAM_H__

#include "stdlib.h"
#include <vector>
#include "fpganetworkinterface.h"

//---------------------------constant and macro define---------------------------//
#define K_VECTOR 16

#define RELU_K_VECTOR 16

#define C_VECTOR 16

#define Q_VECTOR 5 //for resnet50

#define INTERLEAVE 1

#define P_INTERLEAVE 1

#define S_VECTOR 3

#define W_VECTOR ( S_VECTOR + Q_VECTOR - 1 )

#define DDR_BANDWIDTH_IN_BYTES ( 64 )

#define FILTER_DDR_READ_INTERVALS1 ( MYCEIL((C_VECTOR * S_VECTOR), DDR_BANDWIDTH_IN_BYTES) )
#define FILTER_DDR_READ_INTERVALS2 ( MYCEIL((C_VECTOR * 1), DDR_BANDWIDTH_IN_BYTES) )

#define POOL_WINDOW_MAX 3

#define POOL_OFFSET_P (POOL_WINDOW_MAX-1)
#define POOL_OFFSET_Q (POOL_WINDOW_MAX-1)

#define MYCEIL(X, Y) ( ( X - 1 ) / (Y) +1 )
//--------------------------constant and macro define end-------------------------//

typedef struct{
  //int num_block;
  int num_conv;
  int num_layer;
  int num_cnct; //dim
  int num_concate;
  int index_concate;
  int wait_cycle;

  int input_image_channel;
  int input_image_height;
  int input_image_width;
  int first_filter_size;

  int max_output_channel;
  int max_input_channel;

  int max_input_height;
  int max_input_width;
  int max_output_height;
  int max_output_width;

  int max_pool_input_height;
  int max_pool_input_width;

  int max_pool_output_height;
  int max_pool_output_width;

  int max_filter;
  int max_pool_output_wvec;

  int max_pool_window_size;

  int max_pool_pad_size;

  int max_filter_size_1;//channel num of filter size with 3
  int max_filter_size_2;//channel num of filter size with 1

  int max_bias_size;
  // //multi-subnet infor reserved
  // int num_subnet;
} a_frame;

// typedef struct{
//   //for data layer
//   //int input_image_size[3]; //C, H, W
//   //multi-subnet info reserved
//   int conv_num_subnet;
//   int layer_num_subnet;
//   int conv_offset_subnet;
// } subnet_frame;

typedef struct{
  bool enable_info[4];//conv, relu, pool, batchnorm for each conv
  int base_num[4];//kCacheReadBase, kCacheWriteBase, kDDRReadBase, kDDRWriteBase
  bool mem_write_enable[2];//kCacheWriteEnable, kDDRWriteEnable
  bool specific_struct_enable[6];//kEndPoolEnable, kAdditionEnable, kBranchTail, kAdditionReluEnable, kIpoolEnable, kBiasEnable
  int concate_offset[2]; //index 0 is start, index 1 is end
  //int conv_l_index;//? keep or ignore
  //int pool_l_index;
  //int cnct_index; //don't needed any more based the sequencial conv ordering principal from left to right and up to down
  //bool cnct; //? keep or ignore

} b_frame;

// typedef struct {
//   //enum block_type { CONV=0, POOL };
//   bool pool_max;
// } t_pool_block;

typedef struct {
  int input_feature_size[3]; //C, H, W
  int output_feature_size[3]; //N, H+, W+
  int window_size[4]; //N, C, H, W
  int stride_size;
  int pad_size;
} s_block;

typedef struct {
  int input_reader_cyc;

  int end_pool_total_cyc;

  int write_cache_total_cyc;
  std::vector<int> write_cache_cyc;

  int feature_writer_total_cyc;
  std::vector<int> feature_writer_cyc;

  int filter_preload_cyc;
  int filter_reader_conv_total_cyc;
  std::vector<int> filter_reader_conv_cyc;

  int conv_total_cyc;
  std::vector<int> conv_cyc;

  int pool_total_cyc;
  std::vector<int> pool_cyc;
} c_block;

typedef struct {
  b_frame basic_info_frame;
    bool pool_max;
  s_block size_block[2];// conv, pool
  // conv_index info
  int conv_index;//? keep or ignore
  int concate_index;
  int wait_cycle;
  int index_input_layer;
  bool input_concate_connection;
} conv_info_block;

typedef struct {
  int N; //output N of last layer
  int H; //output H of last layer
  int W; //output W of last layer
  int cache_read_base_num;
  int cache_write_base_num;
  bool cache_write_enable;
  bool concate;
  std::vector<int> id;
} block_sibling;

// typedef struct {
//   int conv_count;
//   std::vector<int> id_siblings;
// } id_conv_block;

typedef struct { //module is subnet
//   // output back to data without quantitification
//   float* output_without_q;
//   // output size
//   int output_n;//?
//   int output_c;
//   int output_h;
//   int output_w;
//   // paremeters to caffe
//   // std::vector<param_ly_cf> params;
    a_frame total_framework;
    //subnet_frame subnet_framework;
    std::vector<conv_info_block> conv_blocks;
    c_block cycle_block;
} module_framework_data;


// extern std::vector<std::vector<int> >framework;

// module_framework_data FrameParseStore(std::string filename, std::string netname);
void FrameParseStore(std::string filename, std::string netname, module_framework_data* module_framework);
bool ParamGeneration(module_framework_data* module_framework, std::string netname);

#endif