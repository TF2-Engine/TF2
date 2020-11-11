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
// 1. Performs convolution operations in a shifting manner.
// 2. Sends data in daisy-chains through convolution kernels

inline Mreal MUL(real feature, real filter) {
  if (BIT_IS_SET(filter, 6)) {
    return 0;
  }
 
  if (BIT_IS_SET(filter, 7)) {
    feature = -feature;
  }

  filter = 0x1f & filter;
  Mreal data = feature << filter;
 
  return data;
}

STATIC Mreal DotProduct(DotVector feature_values, DotVector filter_values) {
  int dot_accum = 0; // change from long int to int
  #pragma unroll
  for (int c_inc = 0; c_inc < C_VECTOR; c_inc++)
    dot_accum += MUL(feature_values.v[c_inc], filter_values.v[c_inc]);
  
  return dot_accum;
}

AUTORUN TASK kernel void pe_kernel( void ) {
  int pe_input_data_channel_first_cnt = 0;

  // filter buffer
  DotVector filter_cache[FILTER_CACHE_DEPTH][N_VECTOR][FW_VECTOR];
  BiasBnParam bias_bn_cache[DOUBLE_BUFFER_DIM][N_VECTOR];

  INIT_COUNTER(cycle);

  Mreal result[N_VECTOR][W_VECTOR] = {};
  int cycle_end = FILTER_PRELOAD_CYCLE + CONV_TOTAL_CYCLE;

#if (defined PRINT_PE_INPUT) || (defined PRINT_PE_OUTPUT)
  int debug_cycle = FILTER_PRELOAD_CYCLE + FindConvLayerCycles(NUM_LAYER - 1);
  int debug_range = 1000;
#endif

  #pragma ivdep
  do {
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    
    //printf("pe cycle=%d/%d\n", cycle, cycle_end);

    bool conv_done;

    const PeControlSignal cont = read_channel_intel(pe_control_channel);
    const PeInputFilter pe_filter = read_channel_intel(pe_input_filter_channel);
    const PeInputData pe_in = read_channel_intel(pe_input_data_channel);

    // input feature map data
    DotFeatureVector input_data = pe_in.input_data;
    bool input_data_valid = pe_in.input_data_valid;

    // filter data
    DotFilterVector filter_data = pe_filter.filter_data;
    BiasBnParam bias_bn_data = pe_filter.bias_bn_data;
    bool filter_data_valid = pe_filter.data_valid;
    bool filter_bias_read_page = cont.filter_bias_read_page;
    bool filter_bias_write_page = !filter_bias_read_page;
    int  filter_read_addr = cont.filter_read_addr;
    char filter_read_fw_vec = cont.filter_read_fw_vec;
    int  filter_write_addr = cont.filter_write_addr; 
    int  filter_n = pe_filter.n_inc;
      
    // make sure unnecessary bits are masked off
    filter_n &= BIT_MASK(CLOG2(N_VECTOR));
    filter_read_addr &= BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    filter_read_fw_vec &= BIT_MASK(CLOG2(FW_VECTOR));
    filter_write_addr &= BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    bool conv_start = cont.conv_start;

    conv_done = cont.conv_done[0];
      
    // save filter and bias data for next n_vec  
    if (filter_data_valid) {
      #pragma unroll
      for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
        filter_cache[filter_write_addr][filter_n][fw_inc] = filter_data.v[fw_inc];
      }
      
      // saves bias data and bn parameters to the specific buffer
      if (filter_write_addr == 0 || filter_write_addr == FILTER_CACHE_PAGE_DEPTH) {
        bias_bn_cache[filter_bias_write_page][filter_n] = bias_bn_data;
      }
    }
      
    //
    // read filter and bias data for the current input data
    //
    DotVector filter[N_VECTOR][FW_VECTOR];

    BiasBnParam bias_bn[N_VECTOR];
    #pragma unroll
    for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {
      #pragma unroll
      for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
        filter[n_inc][fw_inc] = filter_cache[filter_read_addr][n_inc][fw_inc];
      }
      bias_bn[n_inc] = bias_bn_cache[filter_bias_read_page][n_inc];
    }
      
    //
    // compute dot product by shifting operation
    //
    Mreal  dot_sum_fw_vec[N_VECTOR][W_VECTOR] = {0};

    if (cont.is_QVECTOR){
      #pragma unroll
      for (int n_inc = 0; n_inc < N_VECTOR; n_inc++)
        #pragma unroll
        for (int ow_inc = 0; ow_inc < OW_VECTOR; ow_inc++) {
          #pragma unroll
          for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
            dot_sum_fw_vec[n_inc][ow_inc] += DotProduct( input_data.v[ow_inc+fw_inc], filter[n_inc][fw_inc]);
#ifdef PRINT_PE_INPUT
            if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) { 
              for (int c_inc = 0; c_inc < C_VECTOR; c_inc++ )
                printf ("PE ow_vec=%d fw_vec=%d c_inc=%d input_data=%d filter=%d cycle=%d\n", ow_inc, fw_inc, c_inc, input_data.v[ow_inc+fw_inc].v[c_inc], filter[n_inc][fw_inc].v[c_inc], cycle);
            }
#endif
        }
      }
    } else {
      #pragma unroll
      for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          dot_sum_fw_vec[n_inc][w_inc] = DotProduct(input_data.v[w_inc], filter[n_inc][filter_read_fw_vec]);
#ifdef PRINT_PE_INPUT
          if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) { 
            for (int c_inc = 0; c_inc < C_VECTOR; c_inc++)
              printf("PE w_inc=%d c_inc=%d fsvec=%d input_data=%d filter=%d cycle=%d\n", w_inc, c_inc, filter_read_fw_vec, input_data.v[w_inc].v[c_inc], filter[n_inc][filter_read_fw_vec].v[c_inc], cycle);
          }
#endif
        }
      }
    }
      
    //
    // add the dot product to the current accumulated value
    //
    #pragma unroll
    for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        Mreal sum = conv_start ? bias_bn[n_inc].bias : result[n_inc][w_inc];
        result[n_inc][w_inc] = sum + dot_sum_fw_vec[n_inc][w_inc];
      }
    }
      
    //
    // send out the result
    // 
    if (input_data_valid && conv_done) {
      PeOutput pe_output;
      
      #pragma unroll
      for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          //float bn_data = (result[n][w_inc] * alpha + beta) * TRANS_INFLAT;  
          //int bn_alpha = (result[n][w_inc] * alpha) >> ALPHA_INFLAT;
          long int bn_alpha_inflat = (long int)result[n_inc][w_inc] * (long int)bias_bn[n_inc].alpha;
          int bn_alpha = bn_alpha_inflat >> ALPHA_INFLAT;
          int bn_data = (((bn_alpha + bias_bn[n_inc].beta) >> (INFLAT - 1)) + 1) >> 1;
          pe_output.data.v[w_inc] = bn_data > REALMAX ? REALMAX : bn_data < REALMIN ? REALMIN : bn_data;
          pe_output.pe_output_relu = cont.pe_output_relu;
#ifdef PRINT_PE_OUTPUT
          if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) 
            printf("PE cycle=%d/%d w_inc=%d result=%d bias_bn.alpha=%d bias_bn.beta=%d pe_output.data.v=%d\n", cycle, cycle_end, w_inc, result[n_inc][w_inc], bias_bn[n_inc].alpha, bias_bn[n_inc].beta, pe_output.data.v[w_inc]);
#endif
        }

        write_channel_intel(pe_output_channel[n_inc], pe_output);
      }
    }

    INCREASE_COUNTER(cycle);
#ifdef ENABLE_INFINITE_LOOPS
    if (COUNTER_DONE(cycle)) { RESET_COUNTER(cycle); }
#endif
  }
#ifdef ENABLE_INFINITE_LOOPS
  while (1);
#else
  while (!COUNTER_DONE(cycle));
#endif
}
