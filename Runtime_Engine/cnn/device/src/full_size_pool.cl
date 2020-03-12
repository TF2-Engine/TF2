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

#include "../../host/inc/cnn.h"

// Functions:
// Performs the full size average pool operation which often locates near the end of a deep neural network.
// TODO: Support the condition NARROW_N_VECTOR != N_VECTOR

TASK kernel void full_size_pool(int frame_num) {
  INIT_COUNTER(frame_index); 
  INIT_COUNTER(frame_cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(h_vec);
  INIT_COUNTER(w_vec);
  
  int layer = 0;

  int frame_cycle_end = END_POOL_TOTAL_CYCLE;

#ifdef PRINT_CYCLE
  printf("END_POOL_TOTAL_CYCLE=%d\frame_index", END_POOL_TOTAL_CYCLE);
#endif

  Sreal result[N_VECTOR] = {0};
  
  #pragma ivdep
  do {
    SET_COUNTER(frame_cycle, frame_cycle_end, 0, frame_cycle_end, 1);
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    
    bool new_layer = false;
    int end_pool_start_cycle = 0;
    int layer_temp = 0;
    
    #pragma unroll
    for (int i = 0; i < NUM_CONVOLUTIONS; i++) {
      if (new_layer) continue;
      if (frame_cycle == end_pool_start_cycle && kEndPoolEnable[i]) {
        layer_temp = i;
        new_layer = true;
      }
      if (kEndPoolEnable[i]) end_pool_start_cycle += FEATURE_WRITER_CYCLE(i);
    }

    if (new_layer) layer = layer_temp;
    
    // write cache start
    //
    // receive pool data
    //
    
    int N = kOutputChannels[layer];
    int H = kPoolOutputHeight[layer];
    int W = kPoolOutputWidth[layer];
    int FH = kFilterSize[layer];
    int P = 1;
    int OW = 1;

    int N_VEC = kNvecEnd[layer];
    int H_VEC = kPoolOutputHeight[layer];
    int W_VEC = kPoolOutputWvecEnd[layer];

    SET_COUNTER(n_vec, kNvecEndMax, 0, N_VEC, 1);
    SET_COUNTER(h_vec, kPoolOutputHeightMax, 0, H_VEC, 1);
    SET_COUNTER(w_vec, W_VEC, 0, W_VEC, 1);

    if (new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(h_vec);
      RESET_COUNTER(w_vec);
      new_layer = false;
    }  
  
    PoolTailOutput end_pool_input = read_channel_intel(end_pool_input_channel);
    PoolTailOutput end_pool_output = PoolTailOutputZero;

    #pragma unroll
    for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        int n = n_vec * N_VECTOR + n_inc;
        int oh = h_vec;
        int ow = w_vec * W_VECTOR + w_inc;
      
        if (n < N && (oh >= 0 && oh < H) && (ow >= 0 && ow < W)) {
          result[n_inc] += end_pool_input.write_data[w_inc][n_inc];
        }
      }
    }    

    if (COUNTER_LAST(h_vec) && COUNTER_LAST(w_vec)) {
         
      #pragma unroll
      for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {
        int n = n_vec * N_VECTOR + n_inc;
        if (n < N) {
          // (1/(7*7))*power(2, 15) = 669
          //float temp = (float)result[n_inc] * 0.02040816;
          //Mreal Mtemp = temp > 0 ? (temp + 0.5) : (temp - 0.5);
          Mreal Mtemp = (((result[n_inc] * 669) >> 14) + 1) >> 1;
          end_pool_output.write_data[0][n_inc] = Mtemp > REALMAX ? REALMAX : Mtemp < REALMIN ? REALMIN : Mtemp;
          int cache_write_offset = kCacheWriteBase[layer];
          int concat_offset = kNStart[layer] / NARROW_N_VECTOR * P * CEIL(OW, W_VECTOR);
          end_pool_output.cache_write_addr = cache_write_offset + concat_offset + n_vec * P * CEIL(OW, W_VECTOR);
          result[n_inc] = 0;
        }
      }
      
      write_channel_intel(end_pool_output_channel, end_pool_output);
    }
    
    INCREASE_COUNTER(w_vec);
    if (COUNTER_DONE(w_vec))  { RESET_COUNTER(w_vec);  INCREASE_COUNTER(h_vec); }
    if (COUNTER_DONE(h_vec))  { RESET_COUNTER(h_vec);  INCREASE_COUNTER(n_vec); }
    if (COUNTER_DONE(n_vec))  { RESET_COUNTER(n_vec);  }
    INCREASE_COUNTER(frame_cycle);

    if (COUNTER_DONE(frame_cycle)) { RESET_COUNTER(frame_cycle); INCREASE_COUNTER(frame_index); }
    
  } while (!COUNTER_DONE(frame_index));

}
