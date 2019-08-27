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

#include "quantization.h"

void read_file(FILE *fp, std::string file_name) {
  fp = fopen(file_name.c_str(), "rb");
  if(!fp) {
    ERROR("Can not open file %s\n", file_name.c_str());
  }
}

void quantization(char *q, float *input, char* file_name) {
  int offset = 0;
  // input data
  FILE *fp;
  fp = fopen(file_name, "r");
  if( !fp ) {
    ERROR("Can not open file %s\n", file_name);
  }
     
  INFO("file_name=%s\n", file_name);

  for( int layer = 0; layer < NUM_LAYER + 1; layer++ ) {
    int conv_layer = layer == 0 ? 0 : layer - 1;
    
    int channel = layer == 0 ? 3 : output_channels[conv_layer];
    for( int c = 0; c < channel; c++ ) {
      int q_value = 0;
      if( ipool_enable[conv_layer] ) {
        q[ offset + c ] = q[ input_layer[conv_layer] * MAX_OUT_CHANNEL + c ];
      } else {
        fscanf( fp, "%d", &q_value );
        q[ offset + c ] = -q_value;
        if( res_tail[conv_layer] ) {
          q[ ( NUM_CONVOLUTIONS + 1 + inception_layer[conv_layer] ) * MAX_OUT_CHANNEL + begin_k[conv_layer] + c ] = -q_value; // 1 is for input data
        }
      }
    }
    offset += MAX_OUT_CHANNEL;
  }
  fclose( fp );
}
