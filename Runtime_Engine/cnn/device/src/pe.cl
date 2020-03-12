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
// 1. Performs convolution operations in a shifting manner.
// 2. Sends data in daisy-chains through convolution kernels
/*
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
}*/

STATIC Mreal DotProduct(uchar16 feature_values, uchar16 filter_values );

// this function is the prototype code for each pe kernels in N_VECTOR pe arrays
void PeFunction(int n_inc) {
  int pe_input_data_channel_first_cnt = 0;

  // filter buffer
  DotVector filter_cache[FILTER_CACHE_DEPTH][NEXT_POWER_OF_2(FW_VECTOR)];
  BiasBnParam bias_bn_cache[DOUBLE_BUFFER_DIM];

  INIT_COUNTER(cycle);

  Mreal result[W_VECTOR]={0};
  int cycle_end = FILTER_PRELOAD_CYCLE + CONV_TOTAL_CYCLE;

#ifdef PRINT_PE_INPUT
  int debug_cycle = FILTER_PRELOAD_CYCLE + find_conv_layer_cycles(NUM_LAYER - 1);
  int debug_range = 100000;
#endif

  #pragma ivdep
  do {
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    bool conv_done;

    PeInputData pe_in;
    PeInputFilter pe_filter;
    PeControlSignal cont;
    
    if (n_inc == 0) {
      cont      = read_channel_intel(pe_control_channel_first);
      pe_filter = read_channel_intel(pe_input_filter_channel_first);
      pe_in     = read_channel_intel(pe_input_data_channel_first);
    } else {                          
      cont      = read_channel_intel(pe_control_channel[n_inc-1]);
      pe_filter = read_channel_intel(pe_input_filter_channel[n_inc-1]);
      pe_in     = read_channel_intel(pe_input_data_channel[n_inc-1]);
    }
    
    write_channel_intel(pe_control_channel[n_inc],      cont);
    write_channel_intel(pe_input_filter_channel[n_inc], pe_filter);
    write_channel_intel(pe_input_data_channel[n_inc],   pe_in);    

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
    if (filter_data_valid && filter_n == n_inc) {
      #pragma unroll
      for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
        filter_cache[filter_write_addr][fw_inc] = filter_data.v[fw_inc];
      }
    
      // saves bias data and bn parameters to the specific buffer
      if (filter_write_addr == 0 || filter_write_addr == FILTER_CACHE_PAGE_DEPTH) {
        bias_bn_cache[filter_bias_write_page] = bias_bn_data;
      }
    }
    
    //
    // read filter and bias data for the current input data
    //
    DotFilterVector filter;

    #pragma unroll
    for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
      filter.v[fw_inc] = filter_cache[filter_read_addr][fw_inc];
    }
 
    BiasBnParam bias_bn = bias_bn_cache[filter_bias_read_page];
    
    //
    // compute dot product by shifting operation
    //
    Mreal  dot_sum_fw_vec[W_VECTOR] = {0};

    uchar16 fea_dat;
    uchar16 fil_dat;

    if (cont.is_QVECTOR){
      #pragma unroll
      for (int ow_inc = 0; ow_inc < OW_VECTOR; ow_inc++) {
        #pragma unroll
        for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
	        fea_dat = *(uchar16 *)(&input_data.v[ow_inc+fw_inc]);
	        fil_dat = *(uchar16 *)(&filter.v[fw_inc]);
          dot_sum_fw_vec[ow_inc] += DotProduct(fea_dat, fil_dat);
#ifdef PRINT_PE_INPUT
          if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) { 
            for (int c_inc = 0; c_inc < C_VECTOR; c_inc++ )
              printf ("PE ow_vec=%d fw_vec=%d c_inc=%d input_data=%d filter=%d cycle=%d\frame_index", ow_inc, fw_inc, c_inc, input_data.v[ow_inc+fw_inc].v[c_inc], filter.v[fw_inc].v[c_inc], cycle);
          }
#endif
        }
      }
    } else {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
	      fea_dat = *(uchar16 *)(&input_data.v[w_inc]);
        fil_dat = *(uchar16 *)(&filter.v[filter_read_fw_vec]);
        dot_sum_fw_vec[w_inc] = DotProduct(fea_dat, fil_dat);
#ifdef PRINT_PE_INPUT
        if (n == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) { 
          for (int c_inc = 0; c_inc < C_VECTOR; c_inc++)
            printf("PE w_inc=%d c_inc=%d fsvec=%d input_data=%d filter=%d cycle=%d\frame_index", w_inc, c_inc, filter_read_fw_vec, input_data.v[w_inc].v[c_inc], filter.v[filter_read_fw_vec].v[c_inc], cycle);
        }
#endif
      }
    }
    
    //
    // add the dot product to the current accumulated value
    //
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      Mreal sum = conv_start ? bias_bn.bias : result[w_inc];
      result[w_inc] = sum + dot_sum_fw_vec[w_inc];
    }
    
    //
    // send out the result
    // 
    if (input_data_valid && conv_done) {
      PeOutput pe_output;
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        //float bn_data = (result[w_inc] * alpha + beta) * TRANS_INFLAT;  
        //int bn_alpha = (result[w_inc] * alpha) >> ALPHA_INFLAT;
        long int bn_alpha_inflat = (long int)result[w_inc] * (long int)bias_bn.alpha;
        int bn_alpha = bn_alpha_inflat >> ALPHA_INFLAT;
        int bn_data = (((bn_alpha + bias_bn.beta) >> (INFLAT - 1)) + 1) >> 1;
        pe_output.data.v[w_inc] = bn_data > REALMAX ? REALMAX : bn_data < REALMIN ? REALMIN : bn_data;
        pe_output.pe_output_relu = cont.pe_output_relu;
#ifdef PRINT_PE_OUTPUT
        if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) 
          printf("PE cycle=%d w_inc=%d result=%d bias_bn.alpha=%d bias_bn.beta=%d pe_output.data.v=%d\frame_index", cycle, w_inc, result[w_inc], bias_bn.alpha, bias_bn.beta, pe_output.data.v[w_inc]);
#endif
      }

      write_channel_intel(pe_output_channel[n_inc], pe_output);
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

AUTORUN TASK kernel void pe_tail() {
  
  while (1) {
    bool valid = false;

    PeInputData pe_input_data = read_channel_nb_intel(pe_input_data_channel[N_VECTOR-1], &valid);

    PeInputFilter pe_input_filter = read_channel_nb_intel(pe_input_filter_channel[N_VECTOR-1], &valid);

    PeControlSignal pe_control = read_channel_nb_intel(pe_control_channel[N_VECTOR-1], &valid);
  }
}

#define PE_KERNEL(X) AUTORUN TASK kernel void pe_kernel_##X() { PeFunction(X); }

PE_KERNEL(0);
#if (N_VECTOR > 1)
PE_KERNEL(1);
#if (N_VECTOR > 2)
PE_KERNEL(2);
#if (N_VECTOR > 3)
PE_KERNEL(3);
#if (N_VECTOR > 4)
PE_KERNEL(4);
#if (N_VECTOR > 5)
PE_KERNEL(5);
#if (N_VECTOR > 6)
PE_KERNEL(6);
#if (N_VECTOR > 7)
PE_KERNEL(7);
#if (N_VECTOR > 8)
PE_KERNEL(8);
#if (N_VECTOR > 9)
PE_KERNEL(9);
#if (N_VECTOR > 10)
PE_KERNEL(10);
#if (N_VECTOR > 11)
PE_KERNEL(11);
#if (N_VECTOR > 12)
PE_KERNEL(12);
#if (N_VECTOR > 13)
PE_KERNEL(13);
#if (N_VECTOR > 14)
PE_KERNEL(14);
#if (N_VECTOR > 15)
PE_KERNEL(15);
#if (N_VECTOR > 16)
PE_KERNEL(16);
#if (N_VECTOR > 17)
PE_KERNEL(17);
#if (N_VECTOR > 18)
PE_KERNEL(18);
#if (N_VECTOR > 19)
PE_KERNEL(19);
#if (N_VECTOR > 20)
PE_KERNEL(20);
#if (N_VECTOR > 21)
PE_KERNEL(21);
#if (N_VECTOR > 22)
PE_KERNEL(22);
#if (N_VECTOR > 23)
PE_KERNEL(23);
#if (N_VECTOR > 24)
PE_KERNEL(24);
#if (N_VECTOR > 25)
PE_KERNEL(25);
#if (N_VECTOR > 26)
PE_KERNEL(26);
#if (N_VECTOR > 27)
PE_KERNEL(27);
#if (N_VECTOR > 28)
PE_KERNEL(28);
#if (N_VECTOR > 29)
PE_KERNEL(29);
#if (N_VECTOR > 30)
PE_KERNEL(30);
#if (N_VECTOR > 31)
PE_KERNEL(31);
#if (N_VECTOR > 32)
PE_KERNEL(32);
#if (N_VECTOR > 33)
PE_KERNEL(33);
#if (N_VECTOR > 34)
PE_KERNEL(34);
#if (N_VECTOR > 35)
PE_KERNEL(35);
#if (N_VECTOR > 36)
PE_KERNEL(36);
#if (N_VECTOR > 37)
PE_KERNEL(37);
#if (N_VECTOR > 38)
PE_KERNEL(38);
#if (N_VECTOR > 39)
PE_KERNEL(39);
#if (N_VECTOR > 40)
PE_KERNEL(40);
#if (N_VECTOR > 41)
PE_KERNEL(41);
#if (N_VECTOR > 42)
PE_KERNEL(42);
#if (N_VECTOR > 43)
PE_KERNEL(43);
#if (N_VECTOR > 44)
PE_KERNEL(44);
#if (N_VECTOR > 45)
PE_KERNEL(45);
#if (N_VECTOR > 46)
PE_KERNEL(46);
#if (N_VECTOR > 47)
PE_KERNEL(47);
#if (N_VECTOR > 48)
PE_KERNEL(48);
#if (N_VECTOR > 49)
PE_KERNEL(49);
#if (N_VECTOR > 50)
PE_KERNEL(50);
#if (N_VECTOR > 51)
PE_KERNEL(51);
#if (N_VECTOR > 52)
PE_KERNEL(52);
#if (N_VECTOR > 53)
PE_KERNEL(53);
#if (N_VECTOR > 54)
PE_KERNEL(54);
#if (N_VECTOR > 55)
PE_KERNEL(55);
#if (N_VECTOR > 56)
PE_KERNEL(56);
#if (N_VECTOR > 57)
PE_KERNEL(57);
#if (N_VECTOR > 58)
PE_KERNEL(58);
#if (N_VECTOR > 59)
PE_KERNEL(59);
#if (N_VECTOR > 60)
PE_KERNEL(60);
#if (N_VECTOR > 61)
PE_KERNEL(61);
#if (N_VECTOR > 62)
PE_KERNEL(62);
#if (N_VECTOR > 63)
PE_KERNEL(63);
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
