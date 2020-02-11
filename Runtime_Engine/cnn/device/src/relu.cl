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

#define FAST_LOOP_BEGIN(var_name, var_capacity, start, end, step) \
  int var_name = (start); \
  int var_name##__end = (end); \
  int var_name##__step = (step); \
  do {

#define FAST_LOOP_END(var_name) \
  var_name += var_name##__step; \
  } while (var_name < var_name##__end)

// Functions:
// Performs ReLU operation.
// TODO: Support the condition NARROW_N_VECTOR != N_VECTOR

TASK kernel void relu(int frame_num) {

  INIT_COUNTER(cycle);
  
  int cycle_end = CONV_TOTAL_WRITE_CACHE;

#ifdef PRINT_CYCLE
  printf("CONV_TOTAL_WRITE_CACHE=%d\n", CONV_TOTAL_WRITE_CACHE);
#endif

  do {
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);

    //printf("RELU cycle=%d/%d\n", cycle, cycle_end);

    FAST_LOOP_BEGIN(nn_vec, CEIL(N_VECTOR, NARROW_N_VECTOR), 0, CEIL(N_VECTOR, NARROW_N_VECTOR), 1) {

      PeOutput pe_output[NARROW_N_VECTOR];
      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
        pe_output[n_inc] = read_channel_intel(pe_drain_output_channel[N_VECTOR -  NARROW_N_VECTOR + n_inc]);
      }
	    
      bool is_QVECTOR = pe_output[0].is_QVECTOR;
      
      ReluOutput relu_output;

      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          //printf("RELU cycle=%d/%d nn_vec=%d n_inc=%d w_inc=%d data=%d\n", cycle, cycle_end, nn_vec, n_inc, w_inc, pe_output[n_inc].data.v[w_inc]);
          relu_output.data[n_inc].v[w_inc] = (!pe_output[n_inc].pe_output_relu || pe_output[n_inc].data.v[w_inc] > 0) ? pe_output[n_inc].data.v[w_inc] : 0;
        }
      }
      write_channel_intel(relu_output_channel, relu_output);

    } FAST_LOOP_END(nn_vec);

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
