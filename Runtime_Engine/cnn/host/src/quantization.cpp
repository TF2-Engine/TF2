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

void Quantization(float (*scale)[2], char* file_name) {
  int offset = 0;
  int pre_branch_l_layer = -1;
  // int res_layer = 0;
  // int pre_res_layer = 0;
  // input data
  FILE *fp;
  fp = fopen(file_name, "rb");
  if (fp==NULL) {
    ERROR("Can not open file %s\n", file_name);
  }
     
  INFO("file_name=%s\n", file_name);

  // for (int layer = 0; layer < NUM_LAYER + 1; layer++) {
  // for (int layer = 0; layer < NUM_CONVOLUTIONS + 1; layer++) {
  for (int layer = 0; layer < NUM_CONVOLUTIONS; layer++) {
    int conv_layer = layer == 0 ? 0 : layer - 1;
    
    int channel = layer == 0 ? 3 : kOutputChannels[conv_layer];
    // for (int c = 0; c < channel; c++) {
      float scale_value = 0.0;
      // float scale_value;
      if (kIpoolEnable[layer]) {
        // q[offset + c] = q[ kInputLayer[conv_layer] * MAX_OUT_CHANNEL + c ];
        // scale[layer][0] = scale[kInputLayer[conv_layer]][0];
        // scale[layer][1] = scale[kInputLayer[conv_layer]][1];
        // scale[layer][2] = scale[kInputLayer[conv_layer]][2];
        scale[layer][0] = scale[kInputLayer[conv_layer]][1];
        scale[layer][1] = scale[kInputLayer[conv_layer]][1];
      } else {
        // fscanf(fp, "%d", &scale_value);
        // q[offset + c] = -q_value;
        // fscanf(fp, "%f", &scale_value);

        // fread(&scale_value, sizeof(float), 1, fp);
        // scale[layer][0] = scale_value;
        // // fscanf(fp, "%f", &scale_value);
        // fread(&scale_value, sizeof(float), 1, fp);
        // scale[layer][1] = scale_value;

        // if(layer == 0)
        // {
        //   fread(&scale_value, sizeof(float), 1, fp);
        //   scale[layer][0] = scale_value;
        //   // fscanf(fp, "%f", &scale_value);
        //   fread(&scale_value, sizeof(float), 1, fp);
        //   scale[layer][1] = scale_value;
        // }
        // else
        // {
        //   fread(&scale_value, sizeof(float), 1, fp);
        //   scale[layer][1] = scale_value;
        //   scale[layer][0] = scale[layer - 1][1];
        // }  
        if((!kDDRWriteEnable[layer])&&(!kAdditionEnable[layer]))
        {
            fread(&scale_value, sizeof(float), 1, fp);
            scale[layer][0] = scale_value;

            if(layer < NUM_CONVOLUTIONS - 1)
            {
              fread(&scale_value, sizeof(float), 1, fp);
              scale[layer][1] = scale_value;
            }
            else
            {
              scale[layer][1] = 1.0;
            }
            // fscanf(fp, "%f", &scale_value);

            // fread(&scale_value, sizeof(float), 1, fp);
            // scale[layer][1] = scale_value;

            // scale[layer][2] = 1.0;
        }
        else if((kDDRWriteEnable[layer])&&(!kAdditionEnable[layer]))
        {
            pre_branch_l_layer =  layer;

            fread(&scale_value, sizeof(float), 1, fp);
            scale[layer][0] = scale_value;

            // scale[layer][2] = 1.0;
        }
        else if((kDDRWriteEnable[layer])&&(kAdditionEnable[layer]))
        {
            // res_layer ++;

            fread(&scale_value, sizeof(float), 1, fp);
            scale[layer][0] = scale_value;

            fread(&scale_value, sizeof(float), 1, fp);
            scale[layer][1] = scale_value;
            if(pre_branch_l_layer != -1) 
            {
              scale[pre_branch_l_layer][1] = scale_value;
              pre_branch_l_layer = -1;
            }

            if((layer < NUM_CONVOLUTIONS - 3) && (kDDRWriteEnable[layer + 1] == 0))
            fread(&scale_value, sizeof(float), 1, fp); //for next pre_layer_scale_0, read but without usage

            // scale[layer][2] = (pre_branch_l_layer != -1) ? scale[layer][1] : scale[pre_res_layer][1];

            // pre_res_layer = layer;
            // pre_branch_l_layer = -1;

            // scale[layer][2] = scale[pre_branch_l_layer][1];

            // pre_branch_l_layer = layer;
        }
        
        // pre_branch_l_layer = -1;
        // pre_res_layer = 0;

        if (kBranchTail[conv_layer]) {
          // q[(NUM_CONVOLUTIONS + 1 + kConcatLayer[conv_layer]) * MAX_OUT_CHANNEL + kNStart[conv_layer] + c] = -q_value; // 1 is for input data
          scale[NUM_CONVOLUTIONS + 1 + kConcatLayer[conv_layer]][0] = scale[layer][0]; // 1 is for input data
          scale[NUM_CONVOLUTIONS + 1 + kConcatLayer[conv_layer]][1] = scale[layer][1]; // 1 is for input data
          // scale[NUM_CONVOLUTIONS + 1 + kConcatLayer[conv_layer]][2] = scale[layer][2]; // 1 is for input data
        }

      }
    // }
    // offset += MAX_OUT_CHANNEL;
  }

  #ifdef PRINT_H_FSCALE_INPUT
    for (int layer = 0; layer < NUM_CONVOLUTIONS; layer++) {
    // printf("FBIT SCALE layer=%d scale_0=%f scale_1=%f scale_2=%f\n", layer ,scale[layer][0], scale[layer][1], scale[layer][2]);
    printf("FBIT SCALE layer=%d scale_0=%f scale_1=%f\n", layer ,scale[layer][0], scale[layer][1]);
    }
  #endif
  // res_layer = 0;
  fclose(fp);
}
