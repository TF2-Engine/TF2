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
#include "cycles_computation.h"
#include "tf2_auto_param.h"

//#define PRINT_CYCLE

//reference
int tf2_auto_find_input_reader_cycles(module_framework_data* module_framework) {

  int C = module_framework->conv_blocks[0].size_block[0].input_feature_size[0];
  int H = module_framework->conv_blocks[0].size_block[0].input_feature_size[1];
  int W = module_framework->conv_blocks[0].size_block[0].input_feature_size[2];
  
  return MYCEIL(C, C_VECTOR) * H * MYCEIL(W, W_VECTOR);
}

int tf2_auto_find_feature_writer_cycles(module_framework_data* module_framework, int c_index) {

  int K = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[0];
  int H = module_framework->conv_blocks[c_index].size_block[1].output_feature_size[1]; //pool output image height
  int W = module_framework->conv_blocks[c_index].size_block[1].output_feature_size[2]; //pool output image weight
  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];

  return MYCEIL(K, K_VECTOR)*MYCEIL(K_VECTOR, RELU_K_VECTOR)*P_INTERLEAVE*MYCEIL(H, P_INTERLEAVE)*INTERLEAVE*MYCEIL(W, W_VECTOR*INTERLEAVE);
}

int tf2_auto_find_feature_writer_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  #pragma unroll 
  for( int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++ ) {
    total_cycles += tf2_auto_find_feature_writer_cycles(module_framework, c_index);
  }
  
  return total_cycles;
}

int tf2_auto_find_end_pool_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  #pragma unroll 
  for( int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++ ) {
    if(module_framework->conv_blocks[c_index].basic_info_frame.specific_struct_enable[0])
    total_cycles += tf2_auto_find_feature_writer_cycles(module_framework, c_index);
  }
  
  return total_cycles;
}

int tf2_auto_find_filter_reader_conv_cycles(module_framework_data* module_framework, int c_index) // new c_index with considering of cnct
{
  int C = module_framework->conv_blocks[c_index].size_block[0].input_feature_size[0];// new c_index with considering of cnct
  int K = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[0];
  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];

  int END_CVEC = R == 1 ? MYCEIL(C, C_VECTOR * S_VECTOR) : MYCEIL(C, C_VECTOR); 
  int END_KVEC = MYCEIL(K, K_VECTOR);
  int END_SS = MYCEIL(R, S_VECTOR);
  
  return module_framework->conv_blocks[c_index].basic_info_frame.specific_struct_enable[4] ? 0 : END_KVEC * END_CVEC * R * END_SS * K_VECTOR;
}

// compute how many cycles it takes to read a single convolution layer filter
int tf2_auto_find_filter_reader_conv_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  for(int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++) {
    total_cycles += tf2_auto_find_filter_reader_conv_cycles(module_framework, c_index);
  }

  return total_cycles;
}

//int tf2_auto_find_filter_preload_cycles(int offset_subnet) {
int tf2_auto_find_filter_preload_cycles(module_framework_data* module_framework) {

  int C = module_framework->conv_blocks[0].size_block[0].input_feature_size[0];
  int R = module_framework->conv_blocks[0].size_block[0].window_size[3];

  return (MYCEIL(C, C_VECTOR) * R * MYCEIL(R, S_VECTOR)) * K_VECTOR * FILTER_DDR_READ_INTERVALS1;
}

int tf2_auto_find_conv_cvec_cycles(module_framework_data* module_framework, int c_index) {

  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];
  int C = module_framework->conv_blocks[c_index].size_block[0].input_feature_size[0];

  int CVEC_iter = MYCEIL(C, C_VECTOR);
  int SS_iter = MYCEIL(R, S_VECTOR);
  
  return  CVEC_iter * R * SS_iter * P_INTERLEAVE * INTERLEAVE;
}

int tf2_auto_find_conv_kvec_cycles(module_framework_data* module_framework, int c_index0, int kvec_th,int k_iter) {

  int c_index =c_index0;
  //calculate conv cycles
  int C = module_framework->conv_blocks[c_index].size_block[0].input_feature_size[0];
  int Q = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[2];
  int P = MYCEIL(module_framework->conv_blocks[c_index].size_block[0].output_feature_size[1], module_framework->conv_blocks[c_index].size_block[0].stride_size); //for resnet50 & googlenet
  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];
  int I = module_framework->conv_blocks[c_index].basic_info_frame.specific_struct_enable[4];

  int Q_iter;
  if(R == 1){// R==1
    Q_iter = MYCEIL(Q, (INTERLEAVE * W_VECTOR));
  }else{ 
    Q_iter = MYCEIL(Q, (INTERLEAVE * Q_VECTOR));
  }
  int P_iter = MYCEIL(P, P_INTERLEAVE);
  int PXQ_iter = P_iter * Q_iter;

  int conv_cyc = PXQ_iter * tf2_auto_find_conv_cvec_cycles(module_framework, c_index0);

  if(kvec_th == k_iter-1) c_index = c_index0+1;

  //calculate write filter cycles
  int FILTER_DDR_READ_INTERVALS = R == 1 ? FILTER_DDR_READ_INTERVALS2 : FILTER_DDR_READ_INTERVALS1;
  C = module_framework->conv_blocks[c_index].size_block[0].input_feature_size[0];
  R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];//update R
  I = module_framework->conv_blocks[c_index].basic_info_frame.specific_struct_enable[4];//update I
  int write_cyc = I ? 0 : (R == 1 ? MYCEIL(C, C_VECTOR * S_VECTOR)*K_VECTOR* FILTER_DDR_READ_INTERVALS : MYCEIL(C, C_VECTOR) * R * MYCEIL(R, S_VECTOR)*K_VECTOR* FILTER_DDR_READ_INTERVALS); //for restnet50 

  if((c_index0 == module_framework->total_framework.num_conv - 1)&&(kvec_th == k_iter-1))
  {
        return conv_cyc;
  }
  else
       return conv_cyc>write_cyc ? conv_cyc : write_cyc;
}

int tf2_auto_find_conv_cycles(module_framework_data* module_framework, int c_index) {

  int K = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[0];
  //#ifdef CYCLES_FUNCS_BACK
  int KVEC_iter = MYCEIL(K, K_VECTOR);
  int res = 0;
  int temp[8]={0};

  for(int i =0;i < KVEC_iter;i++)
  {
        int cur =temp[7]+tf2_auto_find_conv_kvec_cycles(module_framework, c_index, i, KVEC_iter);
        for(int j =7;j>0;j--)
           temp[j] = temp[j-1];
        temp[0] = cur;
  }

  for(int i = 0;i<8;i++)
    res+=temp[i];

  return res;
}

// compute the total number of cycles the sequencer will run for
int tf2_auto_find_conv_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  for(int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++) {
    total_cycles += tf2_auto_find_conv_cycles(module_framework, c_index);
  }
  return total_cycles;
}

int tf2_auto_find_conv_write_cache_cycles(module_framework_data* module_framework, int c_index) {

  int Q = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[2];
  int P = MYCEIL(module_framework->conv_blocks[c_index].size_block[0].output_feature_size[1], module_framework->conv_blocks[c_index].size_block[0].stride_size); //for resnet50 & googlenet
  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];
  int K = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[0];
  int KVEC_iter = MYCEIL(K, K_VECTOR);
    
  int O_VECTOR = R == 1 ? W_VECTOR : Q_VECTOR;

  int Q_iter = MYCEIL(Q, O_VECTOR);
  int P_iter = P;
  int PXQ_iter = P_iter * Q_iter;

  return  KVEC_iter * PXQ_iter;
}

int tf2_auto_find_conv_write_cache_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  //#pragma unroll  
  for(int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++) {
    total_cycles += module_framework->conv_blocks[c_index].basic_info_frame.specific_struct_enable[4] ? 0 : tf2_auto_find_conv_write_cache_cycles(module_framework, c_index);
  }
  return total_cycles;
}
int tf2_auto_find_pool_conv_cycles(module_framework_data* module_framework, int c_index) {

  int R = module_framework->conv_blocks[c_index].size_block[0].window_size[3];
  int END_KVEC = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[0];

  int POOL_PAD_H = module_framework->conv_blocks[c_index].size_block[1].pad_size;
  int POOL_PAD_W = module_framework->conv_blocks[c_index].size_block[1].pad_size;

  int END_HH = module_framework->conv_blocks[c_index].size_block[1].input_feature_size[1] + POOL_OFFSET_P;
  int END_WW = module_framework->conv_blocks[c_index].size_block[0].output_feature_size[2] + POOL_OFFSET_Q;

  int kvec_iter = MYCEIL(END_KVEC, K_VECTOR);
  int hvec_iter = MYCEIL(END_HH, P_INTERLEAVE);
  
  int O_VECTOR = R == 1 ? W_VECTOR : Q_VECTOR;
  
  int wvec_iter = MYCEIL(END_WW, O_VECTOR*INTERLEAVE);
   
  int hhvec_iter = P_INTERLEAVE;
  int wwvec_iter = INTERLEAVE;
  int kkvec_iter = MYCEIL(K_VECTOR, RELU_K_VECTOR);

  return kvec_iter * hvec_iter * wvec_iter * hhvec_iter * wwvec_iter * kkvec_iter;
}

int tf2_auto_find_pool_total_cycles(module_framework_data* module_framework) {
  int total_cycles = 0;
  for(int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++) {
    total_cycles += tf2_auto_find_pool_conv_cycles(module_framework, c_index);
      }
  return total_cycles;
}

bool cycles_computation(module_framework_data* module_framework)
{
  module_framework->cycle_block.input_reader_cyc = tf2_auto_find_input_reader_cycles(module_framework);
  module_framework->cycle_block.filter_preload_cyc = tf2_auto_find_filter_preload_cycles(module_framework);
  module_framework->cycle_block.filter_reader_conv_total_cyc = tf2_auto_find_filter_reader_conv_total_cycles(module_framework);
  module_framework->cycle_block.conv_total_cyc = tf2_auto_find_conv_total_cycles(module_framework);
  module_framework->cycle_block.pool_total_cyc = tf2_auto_find_pool_total_cycles(module_framework);
  module_framework->cycle_block.feature_writer_total_cyc = tf2_auto_find_feature_writer_total_cycles(module_framework);
  module_framework->cycle_block.write_cache_total_cyc = tf2_auto_find_conv_write_cache_total_cycles(module_framework);
  module_framework->cycle_block.end_pool_total_cyc = tf2_auto_find_end_pool_total_cycles(module_framework);

  #ifdef PRINT_CYCLE
  printf("tf_auto - input_reader_cycles=%d\n", module_framework->cycle_block.input_reader_cyc);
  printf("tf_auto - filter_preload_cycles=%d\n", module_framework->cycle_block.filter_preload_cyc);
  printf("tf_auto - filter_reader_conv_total_cycles=%d\n", module_framework->cycle_block.filter_reader_conv_total_cyc);
  printf("tf_auto - conv_total_cycles=%d\n", module_framework->cycle_block.conv_total_cyc);
  printf("tf_auto - pool_total_cycles=%d\n", module_framework->cycle_block.pool_total_cyc);
  printf("tf_auto - feature_writer_total_cycles=%d\n", module_framework->cycle_block.feature_writer_total_cyc);
  printf("tf_auto - write_cache_total_cycles=%d\n", module_framework->cycle_block.write_cache_total_cyc);
  printf("tf_auto - end_pool_total_cycles=%d\n", module_framework->cycle_block.end_pool_total_cyc);
  #endif

  for(int c_index = 0; c_index < module_framework->total_framework.num_conv; c_index++) {
      int filter_reader_conv_cycles = tf2_auto_find_filter_reader_conv_cycles(module_framework, c_index);
      int conv_cycles = tf2_auto_find_conv_cycles(module_framework, c_index);
      int pool_conv_cycles = tf2_auto_find_pool_conv_cycles(module_framework, c_index);
      int feature_writer_cycles = tf2_auto_find_feature_writer_cycles(module_framework, c_index);
      int write_cache_cycles = tf2_auto_find_conv_write_cache_cycles(module_framework, c_index);

      module_framework->cycle_block.filter_reader_conv_cyc.push_back(filter_reader_conv_cycles);
      module_framework->cycle_block.conv_cyc.push_back(conv_cycles);
      module_framework->cycle_block.pool_cyc.push_back(pool_conv_cycles);
      module_framework->cycle_block.feature_writer_cyc.push_back(feature_writer_cycles);
      module_framework->cycle_block.write_cache_cyc.push_back(write_cache_cycles);

      #ifdef PRINT_CYCLE
      printf("tf_auto - filter_reader_conv_cycles=%d, c_index=%d\n", module_framework->cycle_block.filter_reader_conv_cyc[c_index], c_index);
      printf("tf_auto - conv_cycles=%d, c_index=%d\n", module_framework->cycle_block.conv_cyc[c_index], c_index);
      printf("tf_auto - pool_cycles=%d, c_index=%d\n", module_framework->cycle_block.pool_cyc[c_index], c_index);
      printf("tf_auto - feature_cycles=%d, c_index=%d\n", module_framework->cycle_block.feature_writer_cyc[c_index], c_index);
      printf("tf_auto - write_cache_cycles=%d, c_index=%d\n", module_framework->cycle_block.write_cache_cyc[c_index], c_index);
      #endif
  }

  return true;   
}

