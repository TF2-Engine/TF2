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
// 1. Computes the writing address of current layer output feature map data and writes the data to DDR or sends the data to retriever. 
// 2. Optional: Computes the reading address of residual layer output feature map which needs to be added to current layer output feature map.
// 3. Performs the addition operation for feature map.
// TODO: Support the condition NARROW_N_VECTOR != N_VECTOR

TASK kernel void feature_writer(int frame_num, global volatile real* restrict feature_ddr, global BiasBnParam* restrict gl_bias_bn) {
// TASK kernel void feature_writer(int frame_num, global volatile real* restrict feature_ddr) {
  INIT_COUNTER(frame_index);
  INIT_COUNTER(cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(h_vec);
  INIT_COUNTER(w_vec);
  INIT_COUNTER(nn_vec);

  int layer = 0;

#ifdef PRINT_CYCLE
  printf("FEATURE_WRITER_TOTAL_CYCLE=%d\n", FEATURE_WRITER_TOTAL_CYCLE);
#endif

#ifdef PRINT_FEATURE_WRITER_R_CHANNEL_COUNT
  int read_channel_count = 0;
#endif

  #pragma ivdep
  do {
    int cycle_end = FEATURE_WRITER_TOTAL_CYCLE;    
    bool new_layer = false;
    int feature_writer_start_cycle = 0;
    int layer_temp = 0;
    #pragma unroll
    for (int i = DEVICE_START_LAYER; i < DEVICE_END_LAYER; i++) {
      if (new_layer) continue;
      if (cycle == feature_writer_start_cycle) {
        layer_temp = i;
        new_layer = true;
      }
      feature_writer_start_cycle += FEATURE_WRITER_CYCLE(i);
#ifdef PRINT_CYCLE
      //if(layer_temp == NUM_LAYER - 1)
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
    SET_COUNTER(nn_vec, NN_VEC, 0, NN_VEC, 1);

    //printf("FEATURE_WRITER cycle=%d/%d\n", cycle, cycle_end);

    if(new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(h_vec);
      RESET_COUNTER(w_vec);
      RESET_COUNTER(nn_vec);

      new_layer = false;
    }
    
    PoolTailOutput pool_tail_output;
    pool_tail_output = read_channel_intel(feature_writer_input_channel);

    #ifdef PRINT_FEATURE_WRITER_R_CHANNEL_COUNT
      if(layer == PRINT_LAYER - 1) printf("FEATURE_WRITER cycle=%d/%d read_channel_count=%d\n", cycle, cycle_end, read_channel_count++);
    #endif
   
    real res_data[W_VECTOR][NARROW_N_VECTOR] = {{0}};

    // read res data
    if (kAdditionEnable[layer]) {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        #pragma unroll
        for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
          int read_res_offset = kDDRReadBase[layer] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
          unsigned long long int addr =
                    (n_vec * NN_VEC + nn_vec) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_inc * NARROW_N_VECTOR +
                    n_inc;

          res_data[w_inc][n_inc] = feature_ddr[read_res_offset + addr];
        }
      }
    } 
   
    int concat_offset = kNStart[layer] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR);

    //int pre_layer = (kResPreLayer[layer] == 0) ? layer : kResPreLayer[layer];
    //float scale_pre =  gl_bias_bn[pre_layer * MAX_BIAS_SIZE].scale[1];
    float scale_cur = gl_bias_bn[layer * MAX_BIAS_SIZE].scale[1];
    // write feature data 
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      #pragma unroll
      for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {       
        int ddr_write_offset = (kDDRWriteBase[layer] + concat_offset) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
        unsigned long long int addr =
                    (n_vec * NN_VEC + nn_vec) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_inc * NARROW_N_VECTOR +
                    n_inc;
                    
        // Sreal addition = pool_tail_output.write_data[w_inc][n_inc] + res_data[w_inc][n_inc];// /res_pre_scale * current_scale for 4bit
        // Sreal addition = pool_tail_output.write_data[w_inc][n_inc] + (res_data[w_inc][n_inc] / scale_pre) * scale_cur;// /res_pre_scale * current_scale for 4bit
        // Sreal addition = pool_tail_output.write_data[w_inc][n_inc] + (res_data[w_inc][n_inc] * (scale_cur / scale_pre));// /res_pre_scale * current_scale for 4bit
        //float addition_temp = pool_tail_output.write_data[w_inc][n_inc] + (res_data[w_inc][n_inc] * (scale_cur / scale_pre));// /res_pre_scale * current_scale for 4bit
        //Sreal addition = (int)(addition_temp > 0 ? addition_temp + 0.5 : addition_temp - 0.5);
        float res = res_data[w_inc][n_inc] * scale_cur;// /res_pre_scale * current_scale for 4bit
        //float res = (res_data[w_inc][n_inc] * (scale_cur / scale_pre));// /res_pre_scale * current_scale for 4bit
        real real_res = (real)(res > 0.f ? res + 0.5f : res - 0.5f);
        real addition = pool_tail_output.write_data[w_inc][n_inc] + real_res;
        //Sreal addition = (int)(addition_temp > 0 ? addition_temp + 0.5 : addition_temp - 0.5);
 #ifdef FEATURE_WRITER_PRINT_INPUT_ENABLE
    if(layer == PRINT_LAYER - 1)
    printf("FEATURE INPUT layer=%d n_vec=%d, h_vec=%d, w_vec=%d, w_inc=%d, n_inc=%d, addition_temp=%f addition=%d, pool_tail_output=%d, res_real=%f, res_data=%d, scale_pre=%f scale_cur=%f addr=%llu, cycle=%d, frame_index=%d\n", layer, n_vec, h_vec, w_vec, w_inc, n_inc, addition_temp, addition, pool_tail_output.write_data[w_inc][n_inc], res_data[w_inc][n_inc] * (scale_cur / scale_pre), res_data[w_inc][n_inc], scale_pre, scale_cur, addr, cycle, frame_index);
#endif       
        // real addition_real = addition > REALMAX ? REALMAX : addition < REALMIN ? REALMIN : addition; // real range
        // real addition_relu = (!kAdditionReluEnable[layer] || addition_real > 0) ? addition_real : 0; // Relu

        real addition_real = kAdditionReluEnable[layer] ? (addition > REALFBITMAX ? REALFBITMAX : addition < REALFBITMIN ? REALFBITMIN : addition) : addition; // real range for bn comparing consideration
        real addition_relu = (!kAdditionReluEnable[layer] || addition_real > 0) ? addition_real : 0; // Relu for bn comparing consideration
        
        pool_tail_output.write_data[w_inc][n_inc] = addition_relu;

#ifdef CONCAT_LAYER_DEBUG
        int output_offset = 0; 
#else
        int output_offset = layer == (NUM_LAYER - 1) ? (OUTPUT_OFFSET + frame_index * OUTPUT_OFFSET) : 0; 
#endif

        if (kDDRWriteEnable[layer] || layer == (NUM_LAYER - 1)) { 
          feature_ddr[ddr_write_offset + output_offset + addr] = addition_relu;
#ifdef FEATURE_WRITER_PRINT_RESULT_ENABLE
    if(layer == PRINT_LAYER - 1)
    printf("FEATURE RESULT layer=%d n_vec=%d, h_vec=%d, w_vec=%d, w_inc=%d, n_inc=%d, addition_relu=%d, feature_result=%d, addr=%llu, ddr_write_offset=%d, output_offset=%d, index=%llu, cycle=%d, frame_index=%d\n", layer, n_vec, h_vec, w_vec, w_inc, n_inc, addition_relu, feature_ddr[ddr_write_offset + output_offset + addr], addr, ddr_write_offset, output_offset, ddr_write_offset + output_offset + addr, cycle, frame_index);
#endif
        }
      }
    }

    int cache_write_offset = kCacheWriteBase[layer];
    int cache_write_addr = cache_write_offset + concat_offset + (n_vec * NN_VEC + nn_vec) * H * CEIL(W, W_VECTOR) + h_vec * CEIL(W, W_VECTOR) + w_vec;
    pool_tail_output.cache_write_addr = cache_write_addr;
     
    if (kCacheWriteEnable[layer] && !kEndPoolEnable[layer]) {
      write_channel_intel(retriever_input_channel, pool_tail_output);
    }

    if (kEndPoolEnable[layer]) {
      write_channel_intel(end_pool_input_channel, pool_tail_output);
    } 
#ifdef FEATURE_WRITER_PRINT_CYCLE_ENABLE
    if(layer==PRINT_LAYER - 1)
    printf("FEATURE WRITER w_vec=%d, h_vec=%d, n_vec=%d, cycle=%d, frame_index=%d\n", w_vec, h_vec, n_vec, cycle, frame_index);
#endif
    
    INCREASE_COUNTER(nn_vec);
    if (COUNTER_DONE(nn_vec)) { RESET_COUNTER(nn_vec); INCREASE_COUNTER(w_vec); }
    if (COUNTER_DONE(w_vec))  { RESET_COUNTER(w_vec);  INCREASE_COUNTER(h_vec); }
    if (COUNTER_DONE(h_vec))  { RESET_COUNTER(h_vec);  INCREASE_COUNTER(n_vec); }
    if (COUNTER_DONE(n_vec))  { RESET_COUNTER(n_vec);  }
    INCREASE_COUNTER(cycle);

    if (COUNTER_DONE(cycle))  { RESET_COUNTER(cycle); INCREASE_COUNTER(frame_index);}
    
  } while (!COUNTER_DONE(frame_index));
#ifdef PRINT_OUT_INFO
 printf("FEATURE WRITER feature writer out\n");
#endif
}
