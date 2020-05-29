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
// Loads the entire input image of the first layer from DDR   

kernel void input_reader(int frame_num, global volatile const real* restrict input_buffer) {
  INIT_COUNTER(frame_index);
  INIT_COUNTER(cycle);
  INIT_COUNTER(c_vec);
  INIT_COUNTER(h);
  INIT_COUNTER(w_vec);
  
  int cycle_end = INPUT_READER_CYCLE;
  
  do {
    int C = kInputChannels[0];
    int H = kInputHeight[0];
    int W = kInputWidth[0];

    int C_VEC = CEIL(kInputChannels[0], C_VECTOR);
    int END_WW = CEIL(kInputWidth[0], W_VECTOR);
    
    SET_COUNTER(frame_index, frame_num, 0, frame_num, 1);
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    SET_COUNTER(c_vec, C_VEC, 0, C_VEC, 1);
    SET_COUNTER(h, H, 0, H, 1);
    SET_COUNTER(w_vec, END_WW, 0, END_WW, 1);

    InputReaderOutput input_reader_output = InputReaderOutputZero;

    // the data layout is similar to that of filter reader
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      #pragma unroll
      for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
        // linear input column
        int w = (w_vec * W_VECTOR + w_inc);

        // input_buffer data layout: input_buffer[C / C_VECTOR][H][W / W_VECTOR][W_VECTOR * C_VECTOR]
        unsigned long long int conv_input_addr =
                    frame_index * C_VEC * H * CEIL(kInputWidth[0], W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                    c_vec * H * CEIL(kInputWidth[0], W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                    h * CEIL(kInputWidth[0], W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                    w_inc * C_VECTOR +
                    c_inc;

        // linear input channels
        int c = c_vec * C_VECTOR + c_inc;

        bool valid = c >= 0 && c < kInputChannels[0] && w < kInputWidth[0];

        // pad input data for quantization
        real conv_input_data = valid ? input_buffer[conv_input_addr] : 0;
        input_reader_output.data[w_inc].v[c_inc] = conv_input_data;
        //printf("cycle=%d/%d cvec=%d h=%d wvec=%d w_inc=%d c_inc=%d addr=%llu data=%d\n", cycle, cycle_end, c_vec, h, w_vec, w_inc, c_inc, conv_input_addr, conv_input_data);
      }
    }
    write_channel_altera(input_reader_output_channel, input_reader_output);

    INCREASE_COUNTER(w_vec);
    if (COUNTER_DONE(w_vec))  { RESET_COUNTER(w_vec); INCREASE_COUNTER(h); }
    if (COUNTER_DONE(h))      { RESET_COUNTER(h); INCREASE_COUNTER(c_vec); }
    if (COUNTER_DONE(c_vec))  { RESET_COUNTER(c_vec); }
    
    INCREASE_COUNTER(cycle);
    if (COUNTER_DONE(cycle)) { INCREASE_COUNTER(frame_index);  RESET_COUNTER(cycle); RESET_COUNTER(c_vec); RESET_COUNTER(h); RESET_COUNTER(w_vec); }

  } while (!COUNTER_DONE(frame_index));
}
