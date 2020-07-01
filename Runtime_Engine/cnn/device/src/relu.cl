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
// Performs ReLU operation.
// TODO: Support the condition NARROW_N_VECTOR != N_VECTOR

TASK kernel void relu(int frame_num) {

  INIT_COUNTER(cycle);
  
  int cycle_end = CONV_TOTAL_WRITE_CACHE;

#ifdef PRINT_CYCLE
  printf("CONV_TOTAL_WRITE_CACHE=%d\n", CONV_TOTAL_WRITE_CACHE);
#endif

#ifdef PRINT_OUTPUT
  int cnt = 0;
#endif

  do {
    //printf("Relu cycle = %d/%d\n", cycle, cycle_end);

    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);

    PeOutput pe_output[NARROW_N_VECTOR];
    #pragma unroll
    for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
      pe_output[n_inc] = read_channel_intel(pe_output_channel[n_inc]);
    }
	  
    //bool not_1x1_filter = pe_output[0].not_1x1_filter;
    
    ReluOutput relu_output;
    
    #pragma unroll
    for(int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        //if ( not_1x1_filter && w_inc >= OW_VECTOR ) continue;
        relu_output.data[n_inc].v[w_inc] = ( !pe_output[n_inc].pe_output_relu || pe_output[n_inc].data.v[w_inc] > 0 ) ? pe_output[n_inc].data.v[w_inc] : 0;
      }
    }

    write_channel_intel(relu_output_channel, relu_output);

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
