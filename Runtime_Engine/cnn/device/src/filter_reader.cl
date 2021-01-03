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
//#include "ihc_apint.h"
// Functions:
// Reads filter data from DDR and sends it to retriever kernel.

TASK kernel void filter_reader(int frame_num, global real* restrict gl_filter, global BiasBnParam* restrict gl_bias_bn) {
  INIT_COUNTER(cycle);
  INIT_COUNTER(frame_index);
  INIT_COUNTER(frame_cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(c_vec);
  INIT_COUNTER(fh_vec);
  INIT_COUNTER(fw_vec);
  INIT_COUNTER(n_inc);

  int layer = 0;

  int write_cache_addr = 0; 

#ifdef PRINT_CYCLE
  printf("FILTER_READER_CONV_TOTAL_CYCLES=%d\n", FILTER_READER_CONV_TOTAL_CYCLE);
#endif

  do {
    int frame_cycle_end = FILTER_READER_CONV_TOTAL_CYCLE;
    int num_total_cycles = frame_cycle_end * frame_num;

    SET_COUNTER(cycle, MAX_COUNTER_CAPACITY, 0, num_total_cycles, 1); 
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1); 
    SET_COUNTER(frame_cycle, frame_cycle_end, 0, frame_cycle_end, 1); 

    bool new_layer = false;

    int conv_start_cycle = 0;
    int layer_temp = 0;
    #pragma unroll
    for(int i = DEVICE_START_LAYER; i < DEVICE_END_LAYER; i++) {
      if(new_layer && !kIpoolEnable[layer_temp]) continue;
      if(frame_cycle == conv_start_cycle) {
        layer_temp = i;
        new_layer = true;
      }
      conv_start_cycle += FILTER_READER_CONV_CYCLE(i);
#ifdef PRINT_CYCLE
      //if(layer_temp == NUM_LAYER - 1)
      printf("FILTER_READER_CONV_CYCLE(%d)=\t%d\n", i, FILTER_READER_CONV_CYCLE(i));
#endif
    }

    if(new_layer) layer = layer_temp;

    int N =  kOutputChannels[layer]; 
    int C = kInputChannels[layer];
    int FH = kFilterSize[layer];
    int S = kFilterSize[layer];
 
    int N_VEC = kNvecEnd[layer];
    int C_VEC = kFilterCvecEnd[layer];
    int FW_VEC = kFWvecEnd[layer];

    SET_COUNTER(n_vec, kNvecEndMax, 0, N_VEC, 1);
    SET_COUNTER(c_vec, kFilterCvecEndMax, 0, C_VEC, 1);
    SET_COUNTER(fh_vec, kFilterSizeMax, 0, FH, 1);
    SET_COUNTER(fw_vec, kFWvecEndMax, 0, FW_VEC, 1);
    SET_COUNTER(n_inc, N_VECTOR, 0, N_VECTOR, 1);

    if (new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(c_vec);
      RESET_COUNTER(fh_vec);
      RESET_COUNTER(fw_vec);
      RESET_COUNTER(n_inc);

      new_layer = false;
    }

    // first cycle in a n_vec
    if (COUNTER_FIRST(n_inc) && COUNTER_FIRST(fw_vec) && COUNTER_FIRST(fh_vec) && COUNTER_FIRST(c_vec)) {
      write_cache_addr = 0;
    }
	
    //
    // read filter data from DDR
    //	
    FilterReaderOutput filter_reader_output;
    
    int n = n_vec * N_VECTOR + n_inc;
    bool n_valid = n < N;
    
    #pragma unroll
    for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
      #pragma unroll
      for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
        int c = FH == 1 ? c_vec * FW_VECTOR * C_VECTOR + fw_inc * C_VECTOR + c_inc : c_vec * C_VECTOR + c_inc; 
        bool valid = (n_valid && c < C);

        int filter_addr =
                layer * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) + // conv_filter_offset
                n_vec * C_VEC * FH * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                c_vec * FH * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                fh_vec * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                fw_vec * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                n_inc * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                fw_inc * C_VECTOR + 
                c_inc;

        real filter_data = valid ? gl_filter[filter_addr] : 0;
        filter_reader_output.filter_data.v[fw_inc].v[c_inc] = filter_data;
        //if (layer == NUM_LAYER - 1) printf("Filter cycle=%d/%d layer=%d n_vec=%d c_vec=%d fh_vec=%d fw_vec=%d n_inc=%d\n addr=%d filter=%d\n", frame_cycle, frame_cycle_end, layer, n_vec, c_vec, fh_vec, fw_vec, n_inc, filter_addr, filter_data);
        #ifdef PRINT_FILTER_READER_INPUT
        if (layer == NUM_LAYER - 1) printf("Filter cycle=%d/%d layer=%d n_vec=%d c_vec=%d fh_vec=%d fw_vec=%d n_inc=%d\n addr=%d filter=%d\n", frame_cycle, frame_cycle_end, layer, n_vec, c_vec, fh_vec, fw_vec, n_inc, filter_addr, filter_data);
        #endif
      }
    }

    int bias_bn_addr = layer * MAX_BIAS_SIZE + n;
    filter_reader_output.bias_bn_data = n_valid ? gl_bias_bn[bias_bn_addr] : BiasBnZero;
    filter_reader_output.cache_addr = write_cache_addr;
    filter_reader_output.n_inc = n_inc;
   
    write_channel_intel(filter_reader_output_channel, filter_reader_output);
    
    if (COUNTER_LAST(n_inc)) {
      write_cache_addr = (write_cache_addr + 1) & BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    }

    INCREASE_COUNTER(n_inc);
    if (COUNTER_DONE(n_inc))  { RESET_COUNTER(n_inc);    INCREASE_COUNTER(fw_vec); }
    if (COUNTER_DONE(fw_vec)) { RESET_COUNTER(fw_vec);   INCREASE_COUNTER(fh_vec); }
    if (COUNTER_DONE(fh_vec)) { RESET_COUNTER(fh_vec);   INCREASE_COUNTER(c_vec);  }
    if (COUNTER_DONE(c_vec))  { RESET_COUNTER(c_vec);    INCREASE_COUNTER(n_vec);  }
    if (COUNTER_DONE(n_vec))  { RESET_COUNTER(n_vec); }

    INCREASE_COUNTER(frame_cycle);
    if (COUNTER_DONE(frame_cycle)) { RESET_COUNTER(frame_cycle); INCREASE_COUNTER(frame_index); }

    INCREASE_COUNTER(cycle);
  } while (!COUNTER_DONE(cycle));
  
}
