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
// 1. Reads feature cache, sends input feature data (or input data for the input layer) and filter weights to pe kernels.
// 2. Reads feature cache, sends input feature data to pool kernel in ipool layer.
// 3. Receives input image data from input_reader kernel.
// 4. Receives filter data from filter_reader kernel.
// 5. Receives output feature data to be stored in feature_cache from feature_writer kernel.

TASK kernel void retriever(int frame_num, global int* restrict sequencer_idle_cycle) {
  // 
  // create feature cache
  //
  // data layout: feature_cache[read/write cache offset][C/C_VECTOR][H][W / W_VECTOR][W_VECTOR][C_VECTOR]
  
  real feature_cache[CACHE_SIZE][NEXT_POWER_OF_2(W_VECTOR)][NEXT_POWER_OF_2(C_VECTOR)];
  
  //
  // calculate large cycle_counter
  //
  int conv_cycle_end = CONV_TOTAL_CYCLE;
  int input_reader_cycle_end = INPUT_READER_CYCLE;
  int filter_preload_cycle_end = FILTER_PRELOAD_CYCLE;

  //
  // create channel receiver
  //
  SequencerOutput sequencer_output_zero = {0};
  
  //
  // wait cycle_counter calculation
  //
  // cycles to wait after each convolution layer finished

  int local_sequencer_idle_cycle[NUM_LAYER];
  #pragma unroll
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    local_sequencer_idle_cycle[layer] = sequencer_idle_cycle[layer];
  }

  int sequencer_idle_cycle_end = 0;
  #pragma unroll
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    sequencer_idle_cycle_end += local_sequencer_idle_cycle[layer];
  }

  int conv_start_cycles[NUM_LAYER];
  int conv_end_cycles[NUM_LAYER];
  int cycle_counter = input_reader_cycle_end + filter_preload_cycle_end;
  #pragma unroll
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    conv_start_cycles[layer] = cycle_counter;
    conv_end_cycles[layer] = cycle_counter + CONV_CYCLE(layer);
    cycle_counter += CONV_CYCLE(layer) + local_sequencer_idle_cycle[layer];
  }
  
  int cycle_end = input_reader_cycle_end + filter_preload_cycle_end + conv_cycle_end + sequencer_idle_cycle_end;
#ifdef PRINT_CYCLE 
  printf("INPUT_READER_CYCLE=%d FILTER_PRELOAD_CYCLE=%d CONV_TOTAL_CYCLE=%d\n", INPUT_READER_CYCLE, FILTER_PRELOAD_CYCLE, CONV_TOTAL_CYCLE);
#endif
  int filter_ddr_read_cycle = 0;

  INIT_COUNTER(frame_cycle);
  INIT_COUNTER(frame_index);

  do {   
    //printf("RETRIEVER frame_cycle=%d/%d\frame_index", frame_cycle, cycle_end);

    SET_COUNTER(frame_cycle, cycle_end, 0, cycle_end, 1);
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    
    //
    // identify working state
    //
    int cycle = frame_cycle;
    
    // ddr input read mode
    bool input_reading = cycle < input_reader_cycle_end;
    
    // filter preload mode
    bool filter_preloading = !input_reading && cycle < (input_reader_cycle_end + filter_preload_cycle_end);
    
    // sequencer idle mode
    bool sequencer_idle = false;
    #pragma unroll
    for (int layer = 0; layer < NUM_LAYER; layer++) {
      if (layer < (NUM_LAYER - 1) && cycle >= conv_end_cycles[layer] && cycle < conv_start_cycles[layer + 1]) {
        sequencer_idle = true;
      }
    }
    
    bool conving = (input_reading == false && filter_preloading == false && sequencer_idle == false);
    
    // reads sequencer output channel
    SequencerOutput sequencer_output = conving ? read_channel_intel(sequencer_output_channel) : sequencer_output_zero;
    
    {
      int C = kInputChannels[sequencer_output.layer];
      int W = kInputWidth[sequencer_output.layer];
      int H = kInputHeight[sequencer_output.layer];
      int W_VEC = kWvecEnd[sequencer_output.layer];
      int FH = kFilterSize[sequencer_output.layer];

      // reads feature cache read base
      int feature_read_base = kCacheReadBase[sequencer_output.layer];

      //
      // Computes the feature_cache addresses to read from
      //

      int feature_read_addr[W_VECTOR]; 
      int feature_read_w_inc[W_VECTOR]; 

      #pragma unroll
      for(int w_inc = 0; w_inc < W_VECTOR; w_inc++) {

        // padding flag
        bool padding = sequencer_output.padding[w_inc];

        int feature_read_w_vec_cursor = sequencer_output.feature_read_w_vec_header; 

        // the feature map is only banked on the W-dimension
        int feature_read_w_inc_cursor = sequencer_output.feature_read_w_inc_header + w_inc; 
        if (feature_read_w_inc_cursor >= W_VECTOR) {
          feature_read_w_inc_cursor -= W_VECTOR;
          feature_read_w_vec_cursor++;
        }
          
        feature_read_addr[w_inc] = padding ? -1 : 
                                feature_read_base +
                                sequencer_output.c_vec * H * W_VEC +
                                sequencer_output.h * W_VEC +
                                feature_read_w_vec_cursor;

        feature_read_w_inc[w_inc] = padding ? W_VECTOR : feature_read_w_inc_cursor; 
      }

      // ipool layer feature_cache reading address
      int ipool_feature_read_addr = 
                             feature_read_base +
                             sequencer_output.feature_read_w_inc_header * H * W_VEC +
                             sequencer_output.h * W_VEC +
                             sequencer_output.feature_read_w_vec_header;

      //
      // Reads the input feature map data to be sent
      //
      
      DotVector feature_disordered_buffer[W_VECTOR];
          
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        #pragma unroll
        for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
          int feature_read_addr_cursor = -1;
          int this_vec = -1;
          #pragma unroll
          for (int w_inc_cursor = 0; w_inc_cursor < W_VECTOR; w_inc_cursor++) {
            if (feature_read_w_inc[w_inc_cursor] == w_inc) {
              feature_read_addr_cursor = feature_read_addr[w_inc_cursor];
            }
          }

          if (kIpoolEnable[sequencer_output.layer]) feature_read_addr_cursor = ipool_feature_read_addr; 
      
          if (conving && feature_read_addr_cursor != -1) {
            feature_disordered_buffer[w_inc].v[c_inc] = feature_cache[feature_read_addr_cursor][w_inc][c_inc];
          }
        }
      }
    
      //
      // Send input feature map data to convolution kernel#0 (data daisy-chains through convolution kernels)
      //
      DotVector feature_ordered_buffer[W_VECTOR];

      // we loaded W_VECTOR data from W_VECTOR banks
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        #pragma unroll
        for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
          // initializes to zero so that padding value is zero
          feature_ordered_buffer[w_inc].v[c_inc] = 0;

          // each data output can come from INPUT_CACHE_BANK_PER_RDDATA banks
          #pragma unroll
          for (int bank_cursor = 0; bank_cursor < INPUT_CACHE_BANK_PER_RDDATA; bank_cursor++) {
            int w_inc_cursor = (w_inc + bank_cursor * (W_VECTOR / INPUT_CACHE_BANK_PER_RDDATA)) % W_VECTOR;
            if (feature_read_w_inc[w_inc] == w_inc_cursor) {
              feature_ordered_buffer[w_inc].v[c_inc] = feature_disordered_buffer[w_inc_cursor].v[c_inc];
            }
          }
        }
      }

      DotFeatureVector pe_input_data;
      
      #pragma unroll
      for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          pe_input_data.v[w_inc].v[c_inc] = feature_ordered_buffer[w_inc].v[c_inc];
          //printf("cycle=%d/%d c_inc=%d w_inc=%d pe_input_data=%d\n", frame_cycle, cycle_end, c_inc, w_inc, pe_input_data.v[w_inc].v[c_inc]);
        }
      }

      if (input_reading == false && sequencer_idle == false) {
        PeInputData         pe_in       = PeInputDataZero;
        PeInputFilter       pe_filter   = PeInputFilterZero;
        PeControlSignal     pe_cont     = PeControlSignalZero;

        // adds input feature map data
        pe_in.input_data = pe_input_data;
        pe_in.input_data_valid = (!filter_preloading) && (!sequencer_output.filter_loading_conv_idle) && (!kIpoolEnable[sequencer_output.layer]);

        // adds PE control
        pe_cont.is_QVECTOR = (FH != 1);
        pe_cont.conv_start = sequencer_output.conv_start;
        pe_cont.conv_done[0] = sequencer_output.conv_done[0];
        pe_cont.pe_output_relu = kReluEnable[sequencer_output.layer];// && sequencer_output.layer != (NUM_LAYER - 1); // Only For Debug!!!

        // add filter data
        FilterReaderOutput filter_data = FilterReaderOutputZero;
        if (filter_preloading || sequencer_output.filter_loading) {
          int FILTER_DDR_READ_STEP = FH != 1 ? FILTER_DDR_READ_STEP1 : FILTER_DDR_READ_STEP2;
          if (filter_ddr_read_cycle == (FILTER_DDR_READ_STEP - 1)) {
            filter_data = read_channel_intel(filter_reader_output_channel);
            pe_filter.data_valid = true;

            filter_ddr_read_cycle = 0;
          } else {
            filter_ddr_read_cycle++;
          }
        }

        bool filter_read_page;
        if (filter_preloading) {
          // when the sequencer starts, it initially sets filter_read_page
          // to 1, so during the filter preload phase we use the other page
          filter_read_page = 0;
        } else {
          filter_read_page = sequencer_output.filter_read_page;
        }
     
        pe_cont.filter_read_addr =
          ((filter_read_page == 0 ? 0 : FILTER_CACHE_PAGE_DEPTH) +
           sequencer_output.filter_read_addr) & BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    
        pe_cont.filter_read_fw_vec = sequencer_output.filter_read_fw_vec; 

        pe_cont.filter_write_addr =   
          ((filter_read_page == 1 ? 0 : FILTER_CACHE_PAGE_DEPTH) +
           filter_data.cache_addr) & BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));

        pe_cont.filter_bias_read_page = filter_read_page;
        
        pe_filter.n_inc = filter_data.n_inc & BIT_MASK(CLOG2(N_VECTOR));
        pe_filter.filter_data = filter_data.filter_data;
        pe_filter.bias_bn_data = filter_data.bias_bn_data;

        write_channel_intel(pe_control_channel,     pe_cont);
        write_channel_intel(pe_input_filter_channel, pe_filter);
        write_channel_intel(pe_input_data_channel,   pe_in);
      }

      if (input_reading == false && sequencer_idle == false && sequencer_output.filter_loading_conv_idle == 
          false && kIpoolEnable[sequencer_output.layer]) {
        ReluOutput ipool_input;
        
        #pragma unroll
        for (int n_inc = 0; n_inc < NARROW_N_VECTOR; n_inc++) {
          #pragma unroll
          for (int q = 0; q < W_VECTOR; q++) {
            ipool_input.data[n_inc].v[q] = feature_disordered_buffer[q].v[n_inc];
#ifdef PRINT_IPOOL_INPUT
            if (sequencer_output.layer == NUM_LAYER - 1) 
              printf("ipool_feature_read_addr=%d n_vec=%d oh_vec=%d ow_vec=%d n_inc=%d q=%d data=%d ipool_channel_cnt=%d frame_cycle=%d\frame_index", ipool_feature_read_addr, sequencer_output.feature_read_w_inc_header, sequencer_output.h, sequencer_output.feature_read_w_vec_header, n_inc, q, ipool_input.data[n_inc].v[q], ipool_channel_cnt, frame_cycle);
#endif            
          }
        }

        write_channel_intel(ipool_channel, ipool_input);
      }

    } // read cache end
    
    { // write cache start

      //
      // Write to cache every cycle if there is data on the input channels
      //

      //
      // receive data (if any)
      //

      InputReaderOutput input_reader_output;
      PoolTailOutput pool_tail_output;
      
      bool pool_tail_data_received = false;
      
      int feature_write_addr = 0;
      real feature_write_data = 0;
      bool feature_writing = false;
      
      if (input_reading) {
        input_reader_output = read_channel_intel(input_reader_output_channel);
      } else {
        pool_tail_output = read_channel_nb_intel(retriever_input_channel, &pool_tail_data_received);
        if(!pool_tail_data_received) pool_tail_output = read_channel_nb_intel(end_pool_output_channel, &pool_tail_data_received);
      } 
  
      //
      // write data to the cache
      //
      #pragma unroll
      for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
        #pragma unroll
        for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
          if (input_reading) {
            feature_write_addr = cycle;
            feature_write_data = input_reader_output.data[w_inc].v[c_inc];
            feature_writing = true;
          } else {
            feature_write_addr = pool_tail_output.cache_write_addr;
            feature_write_data = pool_tail_output.write_data[w_inc][c_inc];
            feature_writing = pool_tail_data_received;
          }

          if(feature_writing) {
            feature_cache[feature_write_addr][w_inc][c_inc] = feature_write_data;
          }
        }
      }
      
    }
    
    INCREASE_COUNTER(frame_cycle);
    if (COUNTER_DONE(frame_cycle)) { RESET_COUNTER(frame_cycle); INCREASE_COUNTER(frame_index); }

  } while (!COUNTER_DONE(frame_index));
}
