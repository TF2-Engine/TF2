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


#ifndef OPENCL
#define OPENCL
#endif

#include "../../shared/inc/cnn.h"

// Functions:
// 1. Computes the writing address of current layer output feature map data and writes the data to DDR or sends the data to retriever. 
// 2. Optional: Computes the reading address of residual layer output feature map which needs to be added to current layer output feature map.
// 3. Performs the addition operation for feature map.
// TODO: Support the condition NARROW_N_VECTOR != N_VECTOR

TASK kernel void feature_writer(int frame_num, global volatile real* restrict feature_ddr) {
  INIT_COUNTER(frame_index);
  INIT_COUNTER(cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(h_vec);
  INIT_COUNTER(w_vec);

  int layer = 0;

#ifdef PRINT_CYCLE
  printf("FEATURE_WRITER_TOTAL_CYCLE=%d\n", FEATURE_WRITER_TOTAL_CYCLE);
#endif

  #pragma ivdep
  do {
    int cycle_end = FEATURE_WRITER_TOTAL_CYCLE;    
    bool new_layer = false;
    int feature_writer_start_cycle = 0;
    int layer_temp = 0;
    #pragma unroll
    for (int i = 0; i < NUM_CONVOLUTIONS; i++) {
      if (new_layer) continue;
      if (cycle == feature_writer_start_cycle) {
        layer_temp = i;
        new_layer = true;
      }
      feature_writer_start_cycle += FEATURE_WRITER_CYCLE(i);
#ifdef PRINT_CYCLE
      printf("FEATURE_WRITER_CYCLE(%d)=\t%d\n", i, FEATURE_WRITER_CYCLE(i));
#endif
    }

    if (new_layer) layer = layer_temp;
    
    int N = kOutputChannels[layer];
    int H = kPoolOutputHeight[layer];
    int W = kPoolOutputWidth[layer];
    int FH = kFilterSize[layer];
    int N_VEC = kNvecEnd[layer];
    int W_VEC = kPoolOutputWvecEnd[layer];

    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    SET_COUNTER(n_vec, kOutputChannelsMax, 0, N_VEC, 1);
    SET_COUNTER(h_vec, kPoolOutputHeightMax, 0, H, 1);
    //SET_COUNTER(w_vec, W_max, 0, CEIL(W, W_VECTOR), 1);
    SET_COUNTER(w_vec, W_VEC, 0, W_VEC, 1);

    if(new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(h_vec);
      RESET_COUNTER(w_vec);

      new_layer = false;
    }
    
    PoolTailOutput pool_tail_output;
    pool_tail_output = read_channel_intel(feature_writer_input_channel);
   
    real res_data[W_VECTOR][NARROW_N_VECTOR] = {{0}};

    // read res data
    if (kAdditionEnable[layer]) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        #pragma unroll
        for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
          int read_res_offset = kDDRReadBase[layer] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
          unsigned long long int addr =
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_inc * NARROW_N_VECTOR +
                    n_inc;

          res_data[w_inc][n_inc] = feature_ddr[read_res_offset + addr];
        }
      }
    } 
   
    int concat_offset = kNStart[layer] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR);
        
    // write feature data 
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {       
        int ddr_write_offset = (kDDRWriteBase[layer] + concat_offset) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
        unsigned long long int addr =
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_inc * NARROW_N_VECTOR +
                    n_inc;
                    
        Sreal addition = pool_tail_output.write_data[w_inc][n_inc] + res_data[w_inc][n_inc];
        real addition_real = addition > REALMAX ? REALMAX : addition < REALMIN ? REALMIN : addition; // real range
        real addition_relu = (!kAdditionReluEnable[layer] || addition_real > 0) ? addition_real : 0; // Relu
        pool_tail_output.write_data[w_inc][n_inc] = addition_relu;

#ifdef CONCAT_LAYER_DEBUG
        int output_offset = 0; 
#else
        int output_offset = layer == (NUM_LAYER - 1) ? (OUTPUT_OFFSET + frame_index * OUTPUT_OFFSET) : 0; 
#endif

        if (kDDRWriteEnable[layer] || layer == (NUM_LAYER - 1)) { 
          feature_ddr[ddr_write_offset + output_offset + addr] = addition_relu;
        }
      }
    }

    int cache_write_offset = kCacheWriteBase[layer];
    int cache_write_addr = cache_write_offset + concat_offset + n_vec * H * CEIL(W, W_VECTOR) + h_vec * CEIL(W, W_VECTOR) + w_vec;
    pool_tail_output.cache_write_addr = cache_write_addr;
     
    if (kCacheWriteEnable[layer] && !kEndPoolEnable[layer]) {
      write_channel_intel(retriever_input_channel, pool_tail_output);
    }

    if (kEndPoolEnable[layer]) {
      write_channel_intel(end_pool_input_channel, pool_tail_output);
    } 
    
    INCREASE_COUNTER(w_vec);
    if (COUNTER_DONE(w_vec))  { RESET_COUNTER(w_vec);  INCREASE_COUNTER(h_vec); }
    if (COUNTER_DONE(h_vec))  { RESET_COUNTER(h_vec);  INCREASE_COUNTER(n_vec); }
    if (COUNTER_DONE(n_vec))  { RESET_COUNTER(n_vec);  }
    INCREASE_COUNTER(cycle);

    if (COUNTER_DONE(cycle))  { RESET_COUNTER(cycle); INCREASE_COUNTER(frame_index);}
    
  } while (!COUNTER_DONE(frame_index));

}
