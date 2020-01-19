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
// 1. Sets the pace of retriever kernel.
// 2. Computes the filter reading address in pe kernel.
// 3. Computes the starting address of h, w_vec and w_inc of current ow_vec block feature map to be sent to pe kernel from feature_cache in retriever kernel. 
// 4. Sends out working state control signals.

TASK kernel void sequencer(int frame_num) {
  INIT_COUNTER(cycle);
  INIT_COUNTER(frame_index);
  INIT_COUNTER(frame_cycle);
  INIT_COUNTER(n_vec);
  INIT_COUNTER(oh_vec);
  INIT_COUNTER(ow_vec);
  INIT_COUNTER(c_vec);
  INIT_COUNTER(fh_vec);
  INIT_COUNTER(fw_vec);

  int layer = 0;

  int filter_read_page = 0; 
  int filter_read_addr = 0;
  int filter_read_fw_vec = 0; 
  int filter_load_cycle = 0;
  int filter_load_cycle_end = 0;
  bool filter_loading_conv_idle = 0;

  int feature_read_w_vec_from_ow_vec = 0;
  int feature_read_w_inc_from_ow_vec = 0;
  int feature_read_w_vec_from_fh_vec = 0;
  int feature_read_w_inc_from_fh_vec = 0;
  int feature_read_w_vec_from_fw_vec = 0;
  int feature_read_w_inc_from_fw_vec = 0;
  
  do {
    int frame_cycle_end = CONV_TOTAL_CYCLE;
    int cycle_end = frame_cycle_end*frame_num;

    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    SET_COUNTER(frame_cycle, frame_cycle_end, 0, frame_cycle_end, 1);
    if (COUNTER_FIRST(frame_cycle)){
      //layer = 0;
      filter_read_page = 0; 
      filter_read_addr = 0; 
      filter_read_fw_vec = 0;
      filter_load_cycle = 0;
      filter_load_cycle_end = 0;
      filter_loading_conv_idle = 0;
      feature_read_w_vec_from_ow_vec = 0;
      feature_read_w_inc_from_ow_vec = 0;
      feature_read_w_vec_from_fh_vec = 0;
      feature_read_w_inc_from_fh_vec = 0;
      feature_read_w_vec_from_fw_vec = 0;
      feature_read_w_inc_from_fw_vec = 0;
    }

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
      conv_start_cycle += CONV_CYCLE(i);
#ifdef PRINT_CYCLE
      printf("CONV_CYCLE(%d)=\t%d\frame_index", i, CONV_CYCLE(i));
#endif
    }

    if (new_layer)  layer = layer_temp;
    
    int N = kOutputChannels[layer];
    int C = kInputChannels[layer];
    int OW = kOutputWidth[layer]; 
    int OH = kOutputHeight[layer];
    int PAD_H = kPadHeight[layer]; 
    int PAD_W = kPadWidth[layer]; 
    int FH = kFilterSize[layer];
    int W = kInputWidth[layer];
    int H = kInputHeight[layer];

    int N_VEC = kNvecEnd[layer]; 
	
    int WOW_VECTOR = FH != 1 ? OW_VECTOR : W_VECTOR;
	
    int WOW_VEC = CEIL(OW, WOW_VECTOR);
    int C_VEC = kCvecEnd[layer];
    int FW_VEC = kFWvecEnd[layer];

    SET_COUNTER(n_vec,  kNvecEndMax,      0, N_VEC,                    1);
    SET_COUNTER(oh_vec, kOutputHeightMax, 0, OH,      kConvStride[layer]);
    SET_COUNTER(ow_vec, kOutputWidthMax,  0, WOW_VEC,                  1);
    SET_COUNTER(c_vec,  kCvecEndMax,      0, C_VEC,                    1);
    SET_COUNTER(fh_vec, kFilterSizeMax,   0, FH,                       1);
    SET_COUNTER(fw_vec, kFWvecEndMax,     0, FW_VEC,                   1);

    if (new_layer) {
      RESET_COUNTER(n_vec);
      RESET_COUNTER(oh_vec);
      RESET_COUNTER(ow_vec);
      RESET_COUNTER(c_vec);
      RESET_COUNTER(fh_vec);
      RESET_COUNTER(fw_vec);

      new_layer = false;
    }

    bool last_n_vec = COUNTER_LAST(n_vec);
    bool last_conv = (layer == (NUM_LAYER - 1));

    int filter_load_size = kFilterLoadSize[layer];
    int filter_load_size_next = last_conv ? 0 : kFilterLoadSize[layer + 1];

    // start of each n_vec
    if (COUNTER_FIRST(fw_vec) && COUNTER_FIRST(fh_vec) && COUNTER_FIRST(c_vec) &&
        COUNTER_FIRST(oh_vec) && COUNTER_FIRST(ow_vec)) {
#ifdef PRINT_SEQUENCER_INDEX
      printf("SEQUENCER frame_index=%d layer=%d n_vec=%d .......................................\n", frame_index, layer, n_vec);
#endif
    
      int load_size =
          last_n_vec == false ? filter_load_size :
          last_n_vec == true && last_conv == false ? filter_load_size_next : 0;

      int FILTER_DDR_READ_STEP = FH == 1 ? FILTER_DDR_READ_STEP2 : FILTER_DDR_READ_STEP1;

      if (COUNTER_FIRST(frame_cycle)) {
        filter_read_page = 1;
        filter_load_cycle_end = load_size * N_VECTOR * FILTER_DDR_READ_STEP;
      } else {
        if (filter_load_cycle < filter_load_cycle_end) {// filter loading not finished
          filter_loading_conv_idle = 1;
        } else { // filter loading finished, change page
          filter_loading_conv_idle = 0 ;
          filter_load_cycle = 0;
          filter_read_page = !filter_read_page;
          filter_load_cycle_end = load_size * N_VECTOR * FILTER_DDR_READ_STEP;
        }
      }
    }

    // start of each oh_vec
    if (COUNTER_FIRST(fw_vec) && COUNTER_FIRST(fh_vec) && COUNTER_FIRST(c_vec) &&
        COUNTER_FIRST(ow_vec)) {

      // increment by OW_VECTOR/W_VECTOR in every iteration
      feature_read_w_vec_from_ow_vec = -WOW_VECTOR / W_VECTOR;
      feature_read_w_inc_from_ow_vec = -PAD_W - (WOW_VECTOR % W_VECTOR);
    }

    // start of each ow_vec
    if (COUNTER_FIRST(fw_vec) && COUNTER_FIRST(fh_vec) && COUNTER_FIRST(c_vec)) {

      feature_read_w_vec_from_ow_vec += WOW_VECTOR / W_VECTOR;
      feature_read_w_inc_from_ow_vec += WOW_VECTOR % W_VECTOR;
	  
      filter_read_addr = 0;
      filter_read_fw_vec = 0;
    }
    
    // start of each fh_vec
    if (COUNTER_FIRST(fw_vec)) {
      feature_read_w_vec_from_fh_vec = feature_read_w_vec_from_ow_vec;
      feature_read_w_inc_from_fh_vec = feature_read_w_inc_from_ow_vec;
    }

    // start of each fw_vec
    int FW1_VECTOR;

    feature_read_w_vec_from_fw_vec = feature_read_w_vec_from_fh_vec;
    FW1_VECTOR = FH != 1 ? FW_VECTOR : 1;
    feature_read_w_inc_from_fw_vec = feature_read_w_inc_from_fh_vec + fw_vec * (FW1_VECTOR % W_VECTOR);

    // Warning: this can be written in a loop
    if (feature_read_w_inc_from_fw_vec >= 10 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 10 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 10;
    }
    if (feature_read_w_inc_from_fw_vec >= 6 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 6 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 6;
    }
    if (feature_read_w_inc_from_fw_vec >= 5 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 5 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 5;
    }
    
    if (feature_read_w_inc_from_fw_vec >= 4 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 4 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 4;
    }
    if (feature_read_w_inc_from_fw_vec >= 3 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 3 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 3;
    }
    if (feature_read_w_inc_from_fw_vec >= 2 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 2 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 2;
    }
    if (feature_read_w_inc_from_fw_vec >= 1 * W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= 1 * W_VECTOR;
      feature_read_w_vec_from_fw_vec += 1;
    }
    
    /*
    while(feature_read_w_inc_from_fw_vec >= W_VECTOR) {
      feature_read_w_inc_from_fw_vec -= W_VECTOR;
      feature_read_w_vec_from_fw_vec += 1;
    }
    */

    SequencerOutput sequencer_output;

    // reset when we start a new set of ow_vec convolutions
    sequencer_output.conv_start = COUNTER_FIRST(c_vec) && COUNTER_FIRST(fh_vec) && COUNTER_FIRST(fw_vec);

    sequencer_output.layer = (uchar)(layer);
    sequencer_output.c_vec = (ushort)(c_vec);

    // filter reading address
    sequencer_output.filter_read_page = filter_read_page;
    sequencer_output.filter_read_addr = filter_read_addr; 
    sequencer_output.filter_read_fw_vec = filter_read_fw_vec;

    if (FH == 1) {
      if (filter_read_fw_vec == (FW_VECTOR - 1)) {
        filter_read_fw_vec = 0;
        filter_read_addr = (filter_read_addr + 1) & BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
      } else {
        filter_read_fw_vec++;
      }
    } else {
      filter_read_addr = (filter_read_addr + 1) & BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    }
   
    sequencer_output.filter_loading = (filter_load_cycle < filter_load_cycle_end);
    filter_load_cycle++;

    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      if (FH != 1 && w_inc >= OW_VECTOR) continue; 

      int ow = ow_vec * WOW_VECTOR + w_inc;
      int oh = oh_vec;


      int h_header = oh - PAD_H; 
      int w_header = ow - PAD_W; 

      #pragma unroll
      for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
        if (FH == 1 && fw_inc >= 1) continue;

	      int fw = fw_vec * FW1_VECTOR + fw_inc;
        int w_inc_cursor = w_inc + fw_inc;

        // current indices in input feature map
        int w = w_header + fw;
        int h = h_header + fh_vec;

        // if indices are outside the boundaries of the image, send pad data
        bool padding = (w < 0 || w >= W || h < 0 || h >= H);

	      int feature_read_w_vec_from_w_fw_inc = feature_read_w_vec_from_fw_vec;
        int feature_read_w_inc_from_w_fw_inc = (feature_read_w_inc_from_fw_vec + w_inc_cursor);
        if (feature_read_w_inc_from_w_fw_inc >= W_VECTOR) {
          feature_read_w_inc_from_w_fw_inc -= W_VECTOR;
          feature_read_w_vec_from_w_fw_inc++;
        }

        sequencer_output.padding[w_inc_cursor] = padding;
        // starting indices of h, w_vec and w_inc of current ow_vec block feature map to be sent to pe kernel from feature_cache in retriever kernel
        if (w_inc_cursor == 0) {
          sequencer_output.h = h;
          sequencer_output.feature_read_w_vec_header = feature_read_w_vec_from_w_fw_inc; 
          sequencer_output.feature_read_w_inc_header = feature_read_w_inc_from_w_fw_inc;
        }
      }

      // convolution done singal
      bool conv_done =
        COUNTER_LAST(c_vec) && COUNTER_LAST(fh_vec) && COUNTER_LAST(fw_vec) && 
        oh < kOutputHeight[layer] &&
        ow < kOutputWidth[layer];
      
      sequencer_output.conv_done[w_inc] = conv_done; 
    }

    if (kIpoolEnable[layer]) {
      sequencer_output.feature_read_w_inc_header = n_vec;
      sequencer_output.h = oh_vec;
      sequencer_output.feature_read_w_vec_header = ow_vec;
    }

    sequencer_output.filter_loading_conv_idle = filter_loading_conv_idle;
    write_channel_intel(sequencer_output_channel, sequencer_output);

    if (!filter_loading_conv_idle) {
      INCREASE_COUNTER(fw_vec);
      if (COUNTER_DONE(fw_vec))  { RESET_COUNTER(fw_vec);  INCREASE_COUNTER(fh_vec); }
      if (COUNTER_DONE(fh_vec))  { RESET_COUNTER(fh_vec);  INCREASE_COUNTER(c_vec);  }
      if (COUNTER_DONE(c_vec))   { RESET_COUNTER(c_vec);   INCREASE_COUNTER(ow_vec); }
      if (COUNTER_DONE(ow_vec))  { RESET_COUNTER(ow_vec);  INCREASE_COUNTER(oh_vec); }
      if (COUNTER_DONE(oh_vec))  { RESET_COUNTER(oh_vec);  INCREASE_COUNTER(n_vec);  }
      if (COUNTER_DONE(n_vec))   { RESET_COUNTER(n_vec);  }
    }
    INCREASE_COUNTER(frame_cycle);
    if (COUNTER_DONE(frame_cycle)) { RESET_COUNTER(frame_cycle); INCREASE_COUNTER(frame_index); }

    INCREASE_COUNTER(cycle);
  } while (!COUNTER_DONE(cycle));
}
