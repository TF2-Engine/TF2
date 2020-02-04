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
// 1. Performs pool operation.
// 2. If there is no pool operation in current layer, the relu kernel output data pass by this kernel. 

TASK kernel void pool(int frame_num) { 

  INIT_COUNTER(frame_index);
  INIT_COUNTER(cycle);
  INIT_COUNTER(n);
  INIT_COUNTER(oh);
  INIT_COUNTER(ow);
  INIT_COUNTER(nn_vec);

  int layer = 0;

  enum {EDGE_H = (POOL_WINDOW_MAX - 1)};
  enum {EDGE_W = (POOL_WINDOW_MAX - 1)};
  enum {WVEC_ITER = CEIL(kOwEndWithOffsetMax, OW_VECTOR)};
  enum {NNVEC_ITER = CEIL(N_VECTOR, NARROW_N_VECTOR)};
  enum {EDGE_H_BUFFER_SIZE = WVEC_ITER * NNVEC_ITER};
  enum {EDGE_W_BUFFER_SIZE = NNVEC_ITER};
  
  ReluChannelVector edge_buffer[EDGE_W][EDGE_W_BUFFER_SIZE]; 
  ReluChannelVector line_buffer[EDGE_H][W_VECTOR][EDGE_H_BUFFER_SIZE]; 
  
  int edge_h_wvec_addr = 0;
  int edge_w_nnvec_addr = 0;

  int cycle_end = POOL_TOTAL_CYCLE;

#ifdef PRINT_CYCLE
  int ipool_channel_cnt = 0;
  printf("POOL_TOTAL_CYCLE=%d\n", POOL_TOTAL_CYCLE);
#endif

  #pragma ivdep  
  do {
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
   
    //printf("POOL cycle=%d/%d\n", cycle, cycle_end);

    bool new_layer = false;
    int start_cycle = 0;
    int layer_temp = 0;
    #pragma unroll
    for (int i = 0; i < NUM_CONVOLUTIONS; i++) {
      if (new_layer) continue;
      if (cycle == start_cycle) {
        layer_temp = i;
        new_layer = true;
      }
      start_cycle += POOL_CYCLE(i);
#ifdef PRINT_CYCLE
      printf("POOL_CYCLE(%d)=\t%d\n", i, POOL_CYCLE(i));
#endif
    }

    if (new_layer) layer = layer_temp;

    int S = kPoolWindow[layer];
    int H = CEIL(kOutputHeight[layer], kConvStride[layer]);
    int W = kOutputWidth[layer];
    int FH = kFilterSize[layer];
    int N = kNEndWithOffset[layer];

    int OH = kOhEndWithOffset[layer];
    int OW = kOwEndWithOffset[layer];

    int WOW_VECTOR = FH != 1 ? OW_VECTOR : W_VECTOR;

    SET_COUNTER(n, kNEndWithOffsetMax, 0, N, N_VECTOR);
    SET_COUNTER(oh, kOhEndWithOffsetMax, 0, OH, 1 );
    SET_COUNTER(ow, kOwEndWithOffsetMax, 0, OW, WOW_VECTOR); 
    SET_COUNTER(nn_vec, N_VECTOR, 0, N_VECTOR, NARROW_N_VECTOR);
 
    if (new_layer) {
      RESET_COUNTER(n);
      RESET_COUNTER(ow);
      RESET_COUNTER(oh);
      RESET_COUNTER(nn_vec);
      new_layer = false;
    }

    	
    ReluChannelVector w_buffer[EDGE_W + W_VECTOR];

    // read data from relu channel
    int N_START = kNStart[layer];
    int N_END = kNEnd[layer];
	
    bool valid_n = (N_START + n) < N_END;
    bool valid_h = oh < H;
    bool valid_w = ow < W;

    if (valid_n && valid_h && valid_w) {
      ReluOutput relu_output;
      
      relu_output = kIpoolEnable[layer] ? read_channel_intel(ipool_channel) : read_channel_intel(relu_output_channel);

      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          if (FH != 1 && w_inc >= OW_VECTOR) continue;

          valid_w = (ow + w_inc) < W;
          if (valid_w) {
            w_buffer[EDGE_W + w_inc].v[n_inc] = relu_output.data[n_inc].v[w_inc];
#ifdef PRINT_POOL_INPUT
            if (layer == NUM_LAYER - 1) printf("POOL layer=%d n=%d oh=%d ow=%d n_inc=%d w_inc=%d data=%d cycle=%d\n", layer, n, oh, ow, n_inc, w_inc, relu_output.data[n_inc].v[w_inc], cycle);
#endif
          } else {
            w_buffer[EDGE_W + w_inc].v[n_inc] = 0;
          }
        } 
      }
#ifdef PRINT_POOL_INPUT
      //if (kIpoolEnable[layer]) ipool_channel_cnt++;
#endif
      } else { 
      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          if (FH != 1 && w_inc >= OW_VECTOR) continue;
          w_buffer[EDGE_W + w_inc].v[n_inc] = 0;
        }
      }
    } 
 
    // get edge_w data and save into w_buffer
    #pragma unroll
    for (int edge_w = 0; edge_w < EDGE_W; edge_w++) {
      if (COUNTER_FIRST(ow)) {
        w_buffer[edge_w] = ReluChannelVectorZero;
      } else {
        w_buffer[edge_w] = edge_buffer[edge_w][edge_w_nnvec_addr];
      }
    }

    // save some of the input data
    #pragma unroll
    for (int edge_w = 0; edge_w < EDGE_W; edge_w++) {
      //edge_buffer[edge_w][edge_w_nnvec_addr] = w_buffer[WOW_VECTOR + edge_w];
      if (FH != 1){
        edge_buffer[edge_w][edge_w_nnvec_addr] = w_buffer[OW_VECTOR + edge_w];
      } else {
        edge_buffer[edge_w][edge_w_nnvec_addr] = w_buffer[W_VECTOR + edge_w];
      }
    }

    bool compute_pool = kPoolEnable[layer];
    
    ReluChannelVector h_buffer[EDGE_H + 1][W_VECTOR];

    // perform width-pooling
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      if (FH != 1 && w_inc >= OW_VECTOR) continue;
      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
        real pool_in[POOL_WINDOW_MAX] = {0};
        real pool_out = 0;

        #pragma unroll
        for (int s_inc = 0; s_inc < POOL_WINDOW_MAX; s_inc++) {
          if (s_inc < S) { 
            pool_in[s_inc] = w_buffer[w_inc + s_inc].v[n_inc];
          }
        }

        // compute output value
        // If the pool window is not equal to 3, you need to modify the below code.
        if (compute_pool) {
          pool_out = max(max(pool_in[0], pool_in[1]), pool_in[2]);
        } else {
          pool_out = pool_in[0];
        }

        h_buffer[EDGE_H][w_inc].v[n_inc] = pool_out;
      }
    }

    // save data into h_buffer
    int edge_h_addr = edge_h_wvec_addr + edge_w_nnvec_addr;
    #pragma unroll
    for (int edge_h = 0; edge_h < EDGE_H; edge_h++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        if (FH != 1 && w_inc >= OW_VECTOR) continue;

        if (COUNTER_FIRST(oh)) {
          h_buffer[edge_h][w_inc] = ReluChannelVectorZero;
        } else {
          h_buffer[edge_h][w_inc] = line_buffer[edge_h][w_inc][edge_h_addr];
        }
      }
    }

    // save data into line_buffer
    #pragma unroll
    for (int edge_h = 0; edge_h < EDGE_H; edge_h++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        if (FH != 1 && w_inc >= OW_VECTOR) continue;
        line_buffer[edge_h][w_inc][edge_h_addr] = h_buffer[1 + edge_h][w_inc];
      }
    }

    PoolOutput pool_output = {{{{0}}}};

    // perform height-pooling
    #pragma unroll
    for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
	      if (FH != 1 && w_inc >= OW_VECTOR) continue;	
        real pool_in[POOL_WINDOW_MAX] = {0};
        real output_value = 0;

        #pragma unroll
        for (int s_inc = 0; s_inc < POOL_WINDOW_MAX; s_inc++) {
          if (s_inc < S) {
            pool_in[s_inc] = h_buffer[s_inc][w_inc].v[n_inc];
          }
        }

        if (compute_pool) {
          output_value = max(max(pool_in[0], pool_in[1]), pool_in[2]);
        } else {
          output_value = pool_in[0]; 
        }

        pool_output.data[n_inc].v[w_inc] = output_value;
#ifdef PRINT_POOL_OUTPUT
        if (layer == NUM_LAYER - 1) printf("frame_index=%d cycle=%d/%d oh=%d ow=%d nn_vec=%d n=%d n_inc=%d w_inc=%d data=%d\n", frame_index, cycle, cycle_end, oh, ow, nn_vec, n, n_inc, w_inc, output_value);
#endif
      }
    }
    
    write_channel_intel(pool_output_channel, pool_output);

    edge_w_nnvec_addr = (COUNTER_LAST(nn_vec) ? 0 : edge_w_nnvec_addr + 1) & BIT_MASK(CLOG2(NNVEC_ITER));

    if (COUNTER_LAST(nn_vec)) {
      edge_h_wvec_addr = (COUNTER_LAST(ow) ? 0 : edge_h_wvec_addr + NNVEC_ITER) & BIT_MASK(CLOG2(WVEC_ITER * NNVEC_ITER));
    }

    INCREASE_COUNTER(nn_vec);
    if (COUNTER_DONE(nn_vec)) { RESET_COUNTER(nn_vec); INCREASE_COUNTER(ow); }
    if (COUNTER_DONE(ow))     { RESET_COUNTER(ow);  INCREASE_COUNTER(oh); }
    if (COUNTER_DONE(oh))     { RESET_COUNTER(oh);  INCREASE_COUNTER(n); }
    if (COUNTER_DONE(n))      { RESET_COUNTER(n);  }

    INCREASE_COUNTER(cycle);
    if (COUNTER_DONE(cycle))  { RESET_COUNTER(cycle); INCREASE_COUNTER(frame_index); }

#ifdef ENABLE_INFINITE_LOOPS
    if (COUNTER_DONE(cycle))  { RESET_COUNTER(cycle); }
#endif
  }
#ifdef ENABLE_INFINITE_LOOPS
  while (1);
#else
  while (!COUNTER_DONE(frame_index));
#endif
}

