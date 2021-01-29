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
  if (!fp) {
    ERROR("Can not open file %s\n", file_name.c_str());
  }
}

void Quantization(char *q, float *input, char* file_name) {
  int offset = 0;
  // input data
  FILE *fp;
  fp = fopen(file_name, "r");
  if (!fp) {
    ERROR("Can not open file %s\n", file_name);
  }
     
  INFO("file_name=%s\n", file_name);

  for (int layer = 0; layer < NUM_LAYER + 1; layer++) {
    int conv_layer = layer == 0 ? 0 : layer - 1;
    
    int channel = layer == 0 ? 3 : kOutputChannels[conv_layer];

    // For yandi network, in layer 24 yandi_q actually has 8 channel data,
    // the other 24 channel data come from layer 12 and layer 18 for channel concatenation.
    if (getNetwork() == kYandi) {
      if (layer == 24) {
        channel = 8;
      }
    }

    for (int c = 0; c < channel; c++) {
      int q_value = 0;
      if (kIpoolEnable[conv_layer]) {
        q[offset + c] = q[ kInputLayer[conv_layer] * MAX_OUT_CHANNEL + c ];
      } else {
        fscanf(fp, "%d", &q_value);
        q[offset + c] = -q_value;
        if (kBranchTail[conv_layer]) {
          q[(NUM_CONVOLUTIONS + 1 + kConcatLayer[conv_layer]) * MAX_OUT_CHANNEL + kNStart[conv_layer] + c] = -q_value; // 1 is for input data
        }
      }
    }
    offset += MAX_OUT_CHANNEL;
  }
  fclose(fp);

  // For yandi network, concate layer12,layer18 and layer24's Q datat o layer24.
  if (getNetwork() == kYandi) {
    for (int c = 0; c < 8; c++) {
      q[24 * MAX_OUT_CHANNEL + 24 + c] = q[24 * MAX_OUT_CHANNEL + c];
      q[24 * MAX_OUT_CHANNEL + 16 + c] = q[18 * MAX_OUT_CHANNEL + c];
    }
    for (int c = 0; c < 16; c++) {
      q[24 * MAX_OUT_CHANNEL  + c] = q[12 * MAX_OUT_CHANNEL + c];
    }
  }
}
