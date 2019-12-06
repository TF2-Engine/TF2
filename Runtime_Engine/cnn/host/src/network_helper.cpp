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

#include "network_helper.h"

void Verify(int n, char *file_name, char *q, real *output) {
#ifdef CONCAT_LAYER_DEBUG
  int output_channel = kNEnd[NUM_LAYER - 1];
#else
  int output_channel = kOutputChannels[NUM_LAYER - 1];
#endif

  int width = kPoolOutputWidth[NUM_LAYER - 1];
  int height = kPoolOutputHeight[NUM_LAYER - 1];

  //int size = 832 * width * height; // for concat layer splitted branch pool output debug
  int size = output_channel * width * height;
  float *expect = (float*)alignedMalloc(sizeof(float) * size);
  if (!expect) {
    ERROR("malloc expect error!\n");
  }

  FILE *fp;
  //std::string file_name = "model/googlenet/pool1_3x3_s2.bin";
  //std::string file_name = "model/googlenet/conv2_3x3_reduce.bin";
  //std::string file_name = "model/googlenet/pool2_3x3_s2.bin";
  //std::string file_name = "model/googlenet/inception_3a_1x1.bin";
  //std::string file_name = "model/googlenet/inception_3a_3x3_reduce.bin";
  //std::string file_name = "model/googlenet/inception_3a_3x3.bin";
  //std::string file_name = "model/googlenet/inception_3a_5x5_reduce.bin";
  //std::string file_name = "model/googlenet/inception_3a_5x5.bin";
  //std::string file_name = "model/googlenet/inception_3a_pool.bin";
  //std::string file_name = "model/googlenet/inception_3a_pool_proj.bin";
  //std::string file_name = "model/googlenet/inception_3a_output.bin";
  //std::string file_name = "model/googlenet/inception_3b_1x1.bin";
  //std::string file_name = "model/googlenet/inception_3b_3x3_reduce.bin";
  //std::string file_name = "model/googlenet/inception_3b_3x3.bin";
  //std::string file_name = "model/googlenet/inception_3b_5x5_reduce.bin";
  //std::string file_name = "model/googlenet/inception_3b_5x5.bin";
  //std::string file_name = "model/googlenet/inception_3b_pool.bin";
  //std::string file_name = "model/googlenet/inception_3b_pool_proj.bin";
  //std::string file_name = "model/googlenet/pool3_3x3_s2.bin";
  //std::string file_name = "model/googlenet/inception_4a_output.bin";
  //std::string file_name = "model/googlenet/inception_4b_5x5.bin";
  //std::string file_name = "model/googlenet/inception_4b_pool.bin";
  //std::string file_name = "model/googlenet/inception_4b_pool_proj.bin";
  //std::string file_name = "model/googlenet/inception_4b_output.bin";
  //std::string file_name = "model/googlenet/inception_4c_output.bin";
  //std::string file_name = "model/googlenet/inception_4d_output.bin";
  //std::string file_name = "model/googlenet/inception_4e_1x1.bin";
  //std::string file_name = "model/googlenet/inception_4e_3x3_reduce.bin";
  //std::string file_name = "model/googlenet/inception_4e_3x3.bin";
  //std::string file_name = "model/googlenet/inception_4e_5x5.bin";
  //std::string file_name = "model/googlenet/inception_4e_pool.bin";
  //std::string file_name = "model/googlenet/inception_4e_pool_proj.bin";
  //std::string file_name = "model/googlenet/inception_4e_output.bin";
  //std::string file_name = "model/googlenet/pool4_3x3_s2.bin";
  //std::string file_name = "model/googlenet/inception_5a_output.bin";
  //std::string file_name = "model/googlenet/inception_5b_output.bin";
  //std::string file_name = "model/googlenet/loss3_classifier.bin";
  //std::string file_name = "model/resnet50/pool1.bin";
  //std::string file_name = "model/resnet50/fc1000.bin";
  //std::string file_name = "model/ResNet50_pruned/fc1000.bin";
  
  fp = fopen(file_name, "r");
  if (!fp) {
    ERROR("fopen file error!\n");
  }
  fread(expect, sizeof(float), size, fp);

  fclose(fp);

  std::string output_file_name = "Lastconv" + std::to_string(n) + ".dat";

  FILE *fp_output;
  fp_output = fopen(output_file_name.c_str(), "wt");
  int H = height;
  int W = width;
  //int offset = kDDRWriteBase[ NUM_LAYER - 1 ] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
#ifdef CONCAT_LAYER_DEBUG
  int output_offset = 0;
  int concat_offset = 0;
#else
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;
  int concat_offset = kNStart[NUM_LAYER - 1] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
#endif

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;
  for (int n = 0; n < output_channel; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int n_vec = n / N_VECTOR;
        int h_vec = h;
        int w_vec = w / W_VECTOR;
        int ww = w - w_vec * W_VECTOR;
        int nn = n - n_vec * N_VECTOR;
        int addr_out = 
                    output_offset + concat_offset + 
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    ww * NARROW_N_VECTOR +
                    nn;;
        //int addr_exp = ( n + kNStart[ NUM_LAYER - 1 ] ) * H * W + h * W + w; // for concat layer splitted branch pool output debug
        int addr_exp = n * H * W + h * W + w;
        float expect_val = expect[addr_exp] * TRANS_INFLAT;
        float check_error=fabs(expect_val-output[addr_out]);
        total_error += check_error;
        total_expect += fabs(expect_val);
        {
          fprintf(fp_output,"error=%.6f expect=%f expect_trans=%.6f output=%.6f addr=%d n=%d h=%d w=%d\n", check_error, expect[addr_exp], expect_val, 1.0*output[addr_out], addr_out, n, h, w);
        }
      }
    }
  }
  fclose(fp_output);

  INFO("Convolution %d compare finished, error=%f\n", NUM_LAYER, total_error / total_expect);
  if (expect) alignedFree(expect);
}

void Evaluation(int n, char* q, real* output, int* top_labels) {
  int output_channel = kOutputChannels[NUM_LAYER - 1];
  int width = 1;
  int height = 1;
  
  int size = output_channel * width * height;

  FILE *fp;
  fp = fopen("Lastconv.dat","wt");
  int H = height;
  int W = width;
  int ddr_write_offset = kDDRWriteBase[NUM_LAYER - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;

  std::vector<StatItem> stat_array;

  float sum_exp = 0;

  for (int n = 0; n < output_channel; n++) {
    int n_vec = n / N_VECTOR;
    int h_vec = 0;
    int w = 0;
    int w_vec = 0;
    int ww = w - w_vec * W_VECTOR;
    int nn = n - n_vec * N_VECTOR;
    int addr_out = 
                ddr_write_offset + output_offset +
                n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                ww * NARROW_N_VECTOR +
                nn;;
    int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
    float trans = 1 << (-current_q); //take care of shortcut

    StatItem temp;
    temp.label = n;
    temp.feature = output[addr_out] / trans;

    sum_exp += exp(temp.feature);

    stat_array.push_back(temp);
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < output_channel - i - 1; j++) {
      if (stat_array[j].feature > stat_array[j + 1].feature) {
        std::swap(stat_array[j], stat_array[j + 1]);
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    top_labels[i] = stat_array[output_channel - i - 1].label;
    INFO("rank=%d\tlabel=%5d\tprobability=%f\n", i, top_labels[i], exp(stat_array[output_channel - i - 1].feature) / sum_exp);
  }

  fclose(fp);
}

void LoadLabel(int Num,int *labels) {
  FILE *fp;
  if ((fp = fopen("label.dat", "r")) == NULL) {
    ERROR("Error in search label.dat\n");
    exit(0);
  }

  for (int i = 0; i < Num; i++) {
    fscanf(fp,"%d",&labels[i]);
  }

  fclose(fp);
}
