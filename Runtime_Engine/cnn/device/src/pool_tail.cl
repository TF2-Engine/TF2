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

#define NN_VEC CEIL(N_VECTOR, NARROW_N_VECTOR)

// Functions:
// Collects the pool kernel output data, and rearranges it to fit feature_writer kernel.

TASK kernel void pool_tail(int frame_num, global volatile real* restrict feature_ddr) {
  INIT_COUNTER(frame_index); 
  INIT_COUNTER(frame_cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(h_vec);
  INIT_COUNTER(w_vec);
  INIT_COUNTER(nn_vec);
  
  int layer = 0;
 
  bool buffer_index[NN_VEC] = {0};
  bool buffer_done_index[NN_VEC] = {0};
  
  real __attribute__((register)) write_data[NN_VEC][2][NEXT_POWER_OF_2(W_VECTOR)][NEXT_POWER_OF_2(C_VECTOR)];
  
  bool odd_even_factor[NN_VEC] = {0};
  bool odd_even_factor_pool[NN_VEC] = {0};
  int start_wvec_data_addr[NN_VEC] = {0};
  int output_linear_w[NN_VEC] = {0};

  do { 
    int frame_cycle_end = POOL_TOTAL_CYCLE;  // * frame_num BUG
    SET_COUNTER(frame_cycle, frame_cycle_end, 0, frame_cycle_end, 1);
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
   
    //printf("POOL_TAIL cycle=%d/%d\n", frame_cycle, frame_cycle_end);

    bool new_layer = false;

    int conv_start_cycle = 0;
    int layer_temp = 0;
    #pragma unroll
    for (int i = 0; i < NUM_CONVOLUTIONS; i++) {
      if (new_layer) continue;
      if (frame_cycle == conv_start_cycle) {
        layer_temp = i;
        new_layer = true;
      }
      conv_start_cycle += POOL_CYCLE(i);
    }

    if (new_layer) layer = layer_temp;

    int P = kPoolOutputHeight[layer];
    int OW = kPoolOutputWidth[layer];
    int FH = kFilterSize[layer];
    int N_VEC = kNEndWithOffset[layer];
    
    int WOW_VECTOR = FH != 1 ? OW_VECTOR : W_VECTOR;

    int H_VEC = kOhEndWithOffset[layer];
    int W_VEC = kOwEndWithOffset[layer];

    SET_COUNTER(n_vec, kNEndWithOffsetMax, 0, N_VEC, N_VECTOR);
    SET_COUNTER(h_vec, kOhEndWithOffsetMax, 0, H_VEC, 1);
    SET_COUNTER(w_vec, kOwEndWithOffsetMax, 0, W_VEC, WOW_VECTOR);
    SET_COUNTER(nn_vec, NN_VEC, 0, NN_VEC, 1);

    if (new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(h_vec);
      RESET_COUNTER(w_vec);
      RESET_COUNTER(nn_vec);
      new_layer = false;
    }

    bool pool_stride_2 = kPoolStride2[layer];
    bool conv_stride_2 = kConvStride[layer] == 2;

    int ow_offset = pool_stride_2 || conv_stride_2 ? (POOL_OFFSET_P + 1) / 2 : POOL_OFFSET_P - kPoolPad[layer];
    
    if (COUNTER_FIRST(w_vec)) {
      odd_even_factor[nn_vec] = 1;
      start_wvec_data_addr[nn_vec] = -ow_offset;
      output_linear_w[nn_vec] = -ow_offset;
      buffer_index[nn_vec] = 0;
      buffer_done_index[nn_vec] = 0;
    } else {
      odd_even_factor[nn_vec] = !odd_even_factor[nn_vec];
    }

    if (COUNTER_FIRST(w_vec)) {
      odd_even_factor_pool[nn_vec] = 0;
    } else if((OW_VECTOR & 0x1)) {
      odd_even_factor_pool[nn_vec] = !odd_even_factor_pool[nn_vec];
    }

    int step = pool_stride_2 || conv_stride_2 ? (WOW_VECTOR + odd_even_factor[nn_vec]) / 2 : WOW_VECTOR;
    
    //
    // write data to the cache
    //
    
    // receive pool data
    PoolOutput pool_output = read_channel_intel(pool_output_channel);

    int w_index_cache[W_VECTOR];
    int buffer_index_cache[W_VECTOR];
    bool is_fliped = false;
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      int w_cursor = w_inc;
      if (pool_stride_2 || conv_stride_2) {
        if ((w_inc - odd_even_factor_pool[nn_vec]) % 2 != 0) {
          w_index_cache[w_inc] = -1;
          buffer_index_cache[w_inc] = -1;
          continue;
        } else {
          w_cursor = (w_inc - odd_even_factor_pool[nn_vec]) / 2;       
        }
      }

      int w_index = start_wvec_data_addr[nn_vec] + w_cursor;

      if (w_cursor < step && w_index >= W_VECTOR) {
        w_index -= W_VECTOR;
        if (!is_fliped) {
          buffer_index[nn_vec] = !buffer_index[nn_vec];
          is_fliped = true;
        }
      }
      if (w_cursor >= step) w_index = -1; //invalid data
      w_index_cache[w_inc] = w_index;
      buffer_index_cache[w_inc] = buffer_index[nn_vec];
      if (w_cursor == (step - 1) && w_index == (W_VECTOR - 1)) buffer_index[nn_vec] = !buffer_index[nn_vec];
    
      printf("POOL_TAIL1 cycle=%d/%d n_vec=%d w_vec=%d h_vec=%d nn_vec=%d w_inc=%d w_index_cache=%d buffer_index_cache=%d\n", frame_cycle, frame_cycle_end, n_vec, w_vec, h_vec, nn_vec, w_inc, w_index_cache[w_inc], buffer_index_cache[w_inc]);
    }

    #pragma unroll
    for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        real data = 0;
        int buffer_addr = -1; // third dimension

        #pragma unroll
        for (int w_cursor = 0; w_cursor < W_VECTOR; w_cursor++) {
          if (w_index_cache[w_cursor] == w_inc) {
            data = pool_output.data[n_inc].v[w_cursor];
            //data = data_cache[c_inc][w_cursor];
            buffer_addr = buffer_index_cache[w_cursor];
          }
        }

        if (buffer_addr != -1) {
          printf("POOL_TAIL3 cycle=%d/%d n_vec=%d w_vec=%d h_vec=%d nn_vec=%d buffer_addr=%d w_inc=%d n_inc=%d data=%d\n", frame_cycle, frame_cycle_end, n_vec, w_vec, h_vec, nn_vec, buffer_addr, w_inc, n_inc, data);
          write_data[nn_vec][buffer_addr][w_inc][n_inc] = data;
        }
      }
    }

    bool buffer_done = false;
    start_wvec_data_addr[nn_vec] += step;
    bool buffer_done_index_now = -1;
    printf("POOL_TAIL2 cycle=%d/%d n_vec=%d w_vec=%d h_vec=%d nn_vec=%d start_wvec_data_addr=%d COUNTER_LAST(w_vec)=%d output_linear_w=%d\n", frame_cycle, frame_cycle_end, n_vec, w_vec, h_vec, nn_vec, start_wvec_data_addr[nn_vec], COUNTER_LAST(w_vec), output_linear_w[nn_vec]);
    if ((start_wvec_data_addr[nn_vec] >= W_VECTOR || COUNTER_LAST(w_vec)) && (output_linear_w[nn_vec] < OW)) {
      buffer_done = true;
      buffer_done_index_now = buffer_done_index[nn_vec];
      start_wvec_data_addr[nn_vec] -= W_VECTOR;
      buffer_done_index[nn_vec] = !buffer_done_index[nn_vec];
    }

    output_linear_w[nn_vec] += step;
    
    bool write_enable_hvec = true; 
    if ((h_vec < (POOL_OFFSET_P - kPoolPad[layer])) || (!pool_stride_2 && (h_vec > (POOL_OFFSET_P - kPoolPad[layer] + P - 1))) || (pool_stride_2 && (h_vec % 2 != 0))) write_enable_hvec = false;
    
    if (buffer_done && write_enable_hvec) {
      PoolTailOutput pool_tail_output = PoolTailOutputZero;

      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        #pragma unroll
        for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
          pool_tail_output.write_data[w_inc][n_inc] = write_data[nn_vec][buffer_done_index_now][w_inc][n_inc];
          printf("POOL_TAIL cycle=%d/%d n_vec=%d w_vec=%d h_vec=%d w_inc=%d n_inc=%d nn_vec=%d index=%d data=%d\n", frame_cycle, frame_cycle_end, n_vec, w_vec, h_vec, w_inc, n_inc, nn_vec, buffer_done_index_now, pool_tail_output.write_data[w_inc][n_inc]);
        }
      }

      write_channel_intel(feature_writer_input_channel, pool_tail_output);
    }

    INCREASE_COUNTER(frame_cycle);
    if (COUNTER_DONE(frame_cycle))  { RESET_COUNTER(frame_cycle); INCREASE_COUNTER(frame_index); }

    INCREASE_COUNTER(nn_vec);
    if (COUNTER_DONE(nn_vec)) { RESET_COUNTER(nn_vec); INCREASE_COUNTER(w_vec); }
    if (COUNTER_DONE(w_vec))  { RESET_COUNTER(w_vec);  INCREASE_COUNTER(h_vec); }
    if (COUNTER_DONE(h_vec))  { RESET_COUNTER(h_vec);  INCREASE_COUNTER(n_vec); }
    if (COUNTER_DONE(n_vec))  { RESET_COUNTER(n_vec); }

#ifdef ENABLE_INFINITE_LOOPS_POOL_TAIL
    if (COUNTER_DONE(frame_index)) { RESET_COUNTER(frame_index); }
#endif
  }
#ifdef ENABLE_INFINITE_LOOPS_POOL_TAIL
  while (1);
#else
  while (!COUNTER_DONE(frame_index));
#endif
}
