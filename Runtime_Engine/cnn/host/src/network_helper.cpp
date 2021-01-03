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

template void SimCNNLayer<int8,float>(int layerStart, int8 *filter_raw, BiasBnParam *bias_bn, float *input_raw, float (*scale)[2], float *sim_output);
template void Verify(int n, char *file_name, float (*scale)[2], real *output);
template void Evaluation(int n, float (*scale)[2], real *output, int* top_labels);
template void Verify(int n, char *file_name, float (*scale)[2], float *output);
template void Evaluation(int n, float (*scale)[2], float *output, int* top_labels);

template<typename T>
void Verify(int n, char *file_name, float (*scale)[2], T *output) {
#ifdef  FBIT_SIMULATION_END_OUTPUT
  int output_layer = NUM_CONVOLUTIONS;
#else
  int output_layer = NUM_LAYER;
#endif
#ifdef CONCAT_LAYER_DEBUG
  int output_channel = kNEnd[output_layer - 1];
#else
  int output_channel = kOutputChannels[output_layer - 1];
#endif

  int width = kPoolOutputWidth[output_layer - 1];
  int height = kPoolOutputHeight[output_layer - 1];

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
  std::string output_file_name_larger_than_one = "Lastconv" + std::to_string(n) + "_larger_than_one.dat";
  std::string output_file_name_larger_than_one_half = "Lastconv" + std::to_string(n) + "_larger_than_one_half.dat";
  std::string output_file_name_larger_than_half_of_realmax = "Lastconv" + std::to_string(n) + "_larger_than_half_of_realmax.dat";

  FILE *fp_output;
  fp_output = fopen(output_file_name.c_str(), "wt");

  FILE *fp_output_larger_than_one;
  fp_output_larger_than_one = fopen(output_file_name_larger_than_one.c_str(), "wt");

  FILE *fp_output_larger_than_one_half;
  fp_output_larger_than_one_half = fopen(output_file_name_larger_than_one_half.c_str(), "wt");

  FILE *fp_output_larger_than_half_of_realmax;
  fp_output_larger_than_half_of_realmax = fopen(output_file_name_larger_than_half_of_realmax.c_str(), "wt");

  int H = height;
  int W = width;
#ifdef CONCAT_LAYER_DEBUG
  int output_offset = 0;
  int concat_offset = 0;
#else
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;
  int concat_offset = kNStart[output_layer - 1] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
#endif
  int ddr_write_offset = kDDRWriteBase[output_layer - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;
  for (int n = 0; n < output_channel; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int n_vec = n / NARROW_N_VECTOR;
        int h_vec = h;
        int w_vec = w / W_VECTOR;
        int ww = w - w_vec * W_VECTOR;
        int nn = n - n_vec * NARROW_N_VECTOR;
        #if (!defined(CHECK_H_SIMULATION_START_OUTPUT) && !defined(FBIT_SIMULATION_END_OUTPUT))
        int addr_out = 
                    output_offset + concat_offset + ddr_write_offset +
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    ww * NARROW_N_VECTOR +
                    nn;
        #endif
        //int addr_exp = ( n + kNStart[ NUM_LAYER - 1 ] ) * H * W + h * W + w; // for concat layer splitted branch pool output debug
        int addr_exp = n * H * W + h * W + w;
        #if (defined(CHECK_H_SIMULATION_START_OUTPUT) || defined(FBIT_SIMULATION_END_OUTPUT))
        int addr_out = addr_exp;
        #endif
        float expect_val = expect[addr_exp];
// #ifdef CONCAT_LAYER_DEBUG
//        int current_q = q[(NUM_CONVOLUTIONS + 1 + kConcatLayer[NUM_LAYER - 1]) * MAX_OUT_CHANNEL + n];
// #else
//        int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
// #endif
        // float trans = 1 << (-current_q); //take care of shortcut

        float trans = scale[output_layer - 1][1];
        // real expect_c = expect_val > 0.0 ? (real)round(expect_val*trans) : 0;
        real expect_c = (real)round(expect_val*trans);
        // real expect_c = (real)(expect_val*trans);
        #ifdef RES_BN_DEBUG
        float check_error=fabs(expect_c-output[addr_out]);
        #else
        float check_error=fabs(expect_val-output[addr_out]);
        #endif
        total_error += check_error;
        // total_expect += fabs(expect_val*trans);
        #ifdef RES_BN_DEBUG
        total_expect += fabs(expect_c);
        #else
        total_expect += fabs(expect_val);
        #endif
        {
          fprintf(fp_output,"error=%.6f expect_val=%f expect_trans=%.6f output=%.6f addr=%d output_offset=%d concat_offset=%d ddr_write_offset=%d n=%d h=%d w=%d\n", check_error, expect_val, trans, 1.0*output[addr_out], addr_out, output_offset, concat_offset, ddr_write_offset, n, h, w);
          if(check_error >= 1)
          {
            fprintf(fp_output_larger_than_one,"error=%.6f expect_val=%f expect_trans=%.6f output=%.6f addr=%d output_offset=%d concat_offset=%d ddr_write_offset=%d n=%d h=%d w=%d\n", check_error, expect_val, trans, 1.0*output[addr_out], addr_out, output_offset, concat_offset, ddr_write_offset, n, h, w);
            if(check_error > 1.5)
            {
              fprintf(fp_output_larger_than_one_half,"error=%.6f expect_val=%f expect_trans=%.6f output=%.6f addr=%d output_offset=%d concat_offset=%d ddr_write_offset=%d n=%d h=%d w=%d\n", check_error, expect_val, trans, 1.0*output[addr_out], addr_out, output_offset, concat_offset, ddr_write_offset, n, h, w);
              if(check_error > (REALFBITMAX*1.0/2))
              {
                fprintf(fp_output_larger_than_half_of_realmax,"error=%.6f expect_val=%f expect_trans=%.6f output=%.6f addr=%d output_offset=%d concat_offset=%d ddr_write_offset=%d n=%d h=%d w=%d\n", check_error, expect_val, trans, 1.0*output[addr_out], addr_out, output_offset, concat_offset, ddr_write_offset, n, h, w);
              }
            }
          }
        }
      }
    }
  }
  fclose(fp_output);
  fclose(fp_output_larger_than_one);
  fclose(fp_output_larger_than_one_half);
  fclose(fp_output_larger_than_half_of_realmax);

  INFO("Convolution %d compare finished, error=%f\n", output_layer, total_error / total_expect);
  if (expect) alignedFree(expect);
}

template<typename T>
void Evaluation(int n, float (*scale)[2], T* output, int* top_labels) {

#ifdef  FBIT_SIMULATION_END_OUTPUT
  int output_layer = NUM_CONVOLUTIONS;
#else
  int output_layer = NUM_LAYER;
#endif

  int output_channel = kOutputChannels[output_layer - 1];
  int width = 1;
  int height = 1;
  
  int size = output_channel * width * height;

  FILE *fp;
  fp = fopen("Lastconv.dat","wt");
  int H = height;
  int W = width;
  int ddr_write_offset = kDDRWriteBase[output_layer - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;

  std::vector<StatItem> stat_array;

  float sum_exp = 0;

  for (int n = 0; n < output_channel; n++) {
    int n_vec = n / N_VECTOR;
    int h_vec = 0;
    int h = 0;
    int w = 0;
    int w_vec = 0;
    int ww = w - w_vec * W_VECTOR;
    int nn = n - n_vec * N_VECTOR;
    #if (!defined(CHECK_H_SIMULATION_START_OUTPUT) && !defined(FBIT_SIMULATION_END_OUTPUT))
    int addr_out = 
                ddr_write_offset + output_offset +
                n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                ww * NARROW_N_VECTOR +
                nn;
    #endif
    // int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
    // float trans = 1 << (-current_q); //take care of shortcut
    #if (defined(CHECK_H_SIMULATION_START_OUTPUT) || defined(FBIT_SIMULATION_END_OUTPUT))
    int addr_exp = n * H * W + h * W + w;
    int addr_out = addr_exp;
    #endif
    float trans = scale[output_layer - 1][1];

    StatItem temp;
    temp.label = n;
    temp.feature = output[addr_out] / trans;

    sum_exp += exp(temp.feature);

    stat_array.push_back(temp);

    printf("Evaluation n=%d, feature=%f, addr_out=%d, trans=%f sum_exp=%f\n", n, stat_array[n].feature, addr_out, trans, sum_exp);
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
    //INFO( "rank=%d\tlabel=%5d\tfeature=%f\n", i, top_labels[i], stat_array[ output_channel - i - 1 ].feature );
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

template<typename T, typename T1>
void SimCNNLayer(int layerStart, T *filter_raw, BiasBnParam *bias_bn, T1 *input_raw, float (*scale)[2], T1 *sim_output){
  // float scale_local = (layerStart != 0) ? scale, 1.0;
  int layer = layerStart;
  float scale_local = layer == 0 ? 32.0 : scale[layer][1];
  
  int C = layer == 0 ? INPUT_IMAGE_C : kInputChannels[layer];
    
  int H = layer == 0 ? INPUT_IMAGE_H : kInputHeight[layer];
  int W = layer == 0 ? INPUT_IMAGE_W : kInputWidth[layer];
  int N = kOutputChannels[layer];
  int PAD = layer == 0 ? INPUT_IMAGE_PAD : kPadWidth[layer];
  int STRIDE = layer == 0 ? INPUT_IMAGE_STRIDE : kConvStride[layer];

  int H_PAD = H + PAD*2;
  int W_PAD = W + PAD*2;

  int H_Conv_OUT = kOutputHeight[layer];
  int W_Conv_OUT = kOutputWidth[layer];

  int FH = layer == 0 ? FIRST_FILTER_SIZE : kFilterSize[layer];
  int FW = layer == 0 ? FIRST_FILTER_SIZE : kFilterSize[layer];

  int H_Pool_Out = kPoolOutputHeight[layer];
  int W_Pool_Out = kPoolOutputWidth[layer];
  int Pool_PAD = layer == 0 ? 1 : kPoolPad[layer];
  int H_Pool_PAD = H_Conv_OUT + Pool_PAD*2;
  int W_Pool_PAD = W_Conv_OUT + Pool_PAD*2;
  int STRIDE_Pool = kPoolStride2[layer] ? 2 : 1;
  //int STRIDE_Pool = kPoolStride2[layer] ? 2 : 1;
  int FPool = kPoolWindow[layer];
  // Add Pad
  // T1 input_raw_pad[C][H_PAD][W_PAD] = {{{0.0}}};
  // T1 * input_raw_pad = new T1[C][H_PAD][W_PAD]{0.0};
  // T1 input_raw_pad[MAX_OUT_CHANNEL][INPUT_IMAGE_H+INPUT_IMAGE_PAD*2][INPUT_IMAGE_W+INPUT_IMAGE_PAD*2] = {{{0.0}}};
  T1 (*input_raw_pad)[INPUT_IMAGE_H+INPUT_IMAGE_PAD*2][INPUT_IMAGE_W+INPUT_IMAGE_PAD*2] = new T1[MAX_OUT_CHANNEL][INPUT_IMAGE_H+INPUT_IMAGE_PAD*2][INPUT_IMAGE_W+INPUT_IMAGE_PAD*2]{{{0.0}}};
  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H; h++){
      for(int w = 0; w < W; w++){
        // input_raw_pad[c][h+PAD][w+PAD] = input_raw[c * H * W + h * W + w];
        input_raw_pad[c][h+PAD][w+PAD] = input_raw[c * H * W + h * W + w];
        #ifdef PRINT_H_INPUT_RAW
        if(layer == PRINT_LAYER - 1)
        printf("Input Raw layer=%d c=%d h=%d w=%d PAD=%d h+PAD=%d w+PAD=%d input_raw=%f input_raw_pad=%f\n", layer, c, h, w, PAD, h+PAD, w+PAD, input_raw[c * H * W + h * W + w], input_raw_pad[c][h+PAD][w+PAD]);
        #endif
      }
    }
  }
  // Conv operation
  // T1 conv_result[N][H_Conv_OUT][W_Conv_OUT];
  // T1 conv_result[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // T1 conv_result[10][10][10];
  // T1 * conv_result =  new T1[N][H_Conv_OUT][W_Conv_OUT];
  T1 (*conv_result)[kOutputHeightMax][kOutputWidthMax]=  new T1[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // int32_t (*conv_result)[kOutputHeightMax][kOutputWidthMax]=  new int32_t[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // T1 conv_result_m[N][C][H_Conv_OUT][W_Conv_OUT];
  // T1 conv_result_m[kOutputChannelsMax][MAX_OUT_CHANNEL][kOutputHeightMax][kOutputWidthMax];
  // T1 conv_result_m[10][10][10][10];
  // T1 * conv_result_m =  new T1[N][C][H_Conv_OUT][W_Conv_OUT];
  // int32_t (*conv_result_m)[MAX_OUT_CHANNEL][kOutputHeightMax][kOutputWidthMax] =  new int32_t[kOutputChannelsMax][MAX_OUT_CHANNEL][kOutputHeightMax][kOutputWidthMax];
  T1 (*conv_result_m)[MAX_OUT_CHANNEL][kOutputHeightMax][kOutputWidthMax] =  new T1[kOutputChannelsMax][MAX_OUT_CHANNEL][kOutputHeightMax][kOutputWidthMax];

  T1 (*conv_result_scale_trans)[kOutputHeightMax][kOutputWidthMax]=  new T1[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // T1 conv_result_m[10][10][10][10];
  int filter_addr_offset = 0;
  int bias_addr_offset = 0;

  int temp = 0;
  float temp1 = 0.0;
  filter_addr_offset = (MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)) * layer;
  bias_addr_offset = MAX_BIAS_SIZE * layer;
  for(int n = 0; n < N; n++){
      // for(int start_c_h = 0,h_conv_r = 0; start_c_h < H_Conv_OUT; h_conv_r++, start_c_h = STRIDE * h_conv_r){
      //   for(int start_c_w = 0,w_conv_r = 0; start_c_w < W_Conv_OUT; w_conv_r++, start_c_w = STRIDE * w_conv_r){
    for(int start_c_h = 0,h_conv_r = 0; (start_c_h + FH) <= H_PAD; h_conv_r++, start_c_h = STRIDE * h_conv_r){
      for(int start_c_w = 0,w_conv_r = 0; (start_c_w + FW) <= W_PAD; w_conv_r++, start_c_w = STRIDE * w_conv_r){
        for(int c = 0; c < C; c++){
          for(int h = start_c_h; (h < start_c_h + FH) && (h < H_PAD); h++){
            for(int w = start_c_w; (w < start_c_w + FW) && (w < W_PAD); w++){
              // conv_result_m[n][c][h_conv_r][w_conv_r] = + (input_raw_pad[c][h][w] * reinterpret_cast<T1>(filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + h * FW + w].value));
              // temp = filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + h * FW + w].value;

              // temp1 += input_raw_pad[c][h][w] * temp;
              // conv_result_m[n][c][h_conv_r][w_conv_r] = temp1;
              // conv_result_m[n][c][h_conv_r][w_conv_r] += input_raw_pad[c][h][w] * temp;
              // conv_result_m[n][c][h_conv_r][w_conv_r] += (input_raw_pad[c][h][w] * scale_local) * filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + h * FW + w].value;
              conv_result_m[n][c][h_conv_r][w_conv_r] += (input_raw_pad[c][h][w] * scale_local) * filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + (h-start_c_h) * FW + (w-start_c_w)].char_value;
          #ifdef PRINT_SIM_PE_OUTPUT
            // printf("SIM_PE OUTPUT n=%d c=%d h=%d w=%d h_conv_r=%d w_conv_r=%d conv_result_m=%d input_raw_pad=%f layer=%d filter_addr_offset=%d filter_raw=%.3f\n", n, c, h, w, h_conv_r, w_conv_r, conv_result_m[n][c][h_conv_r][w_conv_r], input_raw_pad[c][h][w], layer, filter_addr_offset, filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + h * FW + w].value*1.0);
            if(layer == PRINT_LAYER - 1)
              printf("SIM_PE OUTPUT layer=%d n=%d c=%d h=%d w=%d h_conv_r=%d w_conv_r=%d conv_result_m=%d input_raw_pad=%f filter_addr_offset=%d filter_raw=%.3f\n", layer, n, c, h, w, h_conv_r, w_conv_r, conv_result_m[n][c][h_conv_r][w_conv_r], input_raw_pad[c][h][w], filter_addr_offset, filter_raw[filter_addr_offset + n * C * FH * FW + c * FH * FW + h * FW + w].char_value*1.0);
          #endif
            }
          }
          conv_result[n][h_conv_r][w_conv_r] += conv_result_m[n][c][h_conv_r][w_conv_r];
          if(c == C-1)
          {
            conv_result[n][h_conv_r][w_conv_r] += bias_bn[bias_addr_offset + n].bias;
            #ifdef PRINT_SIM_PE_CONV_OUTPUT
              if(layer == PRINT_LAYER - 1)
              printf("SIM_PE OUTPUT layer=%d n=%d h_conv_r=%d w_conv_r=%d bias_addr_offset=%d bias=%d scale_0=%f conv_result=%d\n", layer, n, h_conv_r, w_conv_r, bias_addr_offset,  bias_bn[bias_addr_offset + n].bias, scale[layer][0], conv_result[n][h_conv_r][w_conv_r]);
            #endif
          }
        }
      }
      // conv_result[n][h_conv_r][w_conv_r] = + conv_result_m[n][c][h_conv_r][w_conv_r];
    }
    // conv_result[n][h_conv_r][w_conv_r] = + bias_bn[bias_addr_offset + n].bias;
  }

  for(int n = 0; n < N; n++){
    // for(int start_c_h = 0,h_conv_r = 0; start_c_h < H_Conv_OUT; h_conv_r++, start_c_h = STRIDE * h_conv_r){
    //   for(int start_c_w = 0,w_conv_r = 0; start_c_w < W_Conv_OUT; w_conv_r++, start_c_w = STRIDE * w_conv_r){
    for(int start_c_h = 0,h_conv_r = 0; (start_c_h + FH) <= H_PAD; h_conv_r++, start_c_h = STRIDE * h_conv_r){
      for(int start_c_w = 0,w_conv_r = 0; (start_c_w + FW) <= W_PAD; w_conv_r++, start_c_w = STRIDE * w_conv_r){
          conv_result_scale_trans[n][h_conv_r][w_conv_r] = (conv_result[n][h_conv_r][w_conv_r]*1.0)/scale[layer][0];
          #ifdef PRINT_SIM_PE_CONV_SCALE_OUTPUT
            if(layer == PRINT_LAYER - 1)
            printf("SIM_PE OUTPUT layer=%d n=%d h_conv_r=%d w_conv_r=%d scale_0=%f conv_result=%d conv_result_scale_trans=%f\n", layer, n, h_conv_r, w_conv_r, scale[layer][0], conv_result[n][h_conv_r][w_conv_r], conv_result_scale_trans[n][h_conv_r][w_conv_r]);
          #endif
      }
    }
  }

  // T1 bn_result[N][H_Conv_OUT][W_Conv_OUT];
  // T1 ReLu_result[N][H_Conv_OUT][W_Conv_OUT];
  // T1 Pool_result[N][H_Pool_Out][W_Pool_Out];

  // T1 bn_result[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // T1 ReLu_result[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // T1 Pool_result[kOutputChannelsMax][kPoolOutputHeightMax][kPoolOutputWidthMax];
  // T1 bn_result[10][10][10];
  // T1 ReLu_result[10][10][10];
  // T1 Pool_result[10][10][10];

  // T1 *bn_result = new T1[N][H_Conv_OUT][W_Conv_OUT];
  // T1 *ReLu_result= new T1[N][H_Conv_OUT][W_Conv_OUT];
  // T1 *Pool_result= new T1[N][H_Pool_Out][W_Pool_Out];

  T1 (*bn_result)[kOutputHeightMax][kOutputWidthMax] = new T1[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // int32_t (*bn_result)[kOutputHeightMax][kOutputWidthMax] = new int32_t[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];

  T1 (*ReLu_result)[kOutputHeightMax][kOutputWidthMax] = new T1[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  T1 (*ReLu_result_scale_trans)[kOutputHeightMax][kOutputWidthMax] = new T1[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  T1 (*ReLu_result_scale_trans_pad)[kOutputHeightMax+1*2][kOutputWidthMax+1*2] = new T1[kOutputChannelsMax][kOutputHeightMax+1*2][kOutputWidthMax+1*2]{{{0.0}}};
  // int32_t (*ReLu_result)[kOutputHeightMax][kOutputWidthMax] = new int32_t[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];
  // int32_t (*ReLu_result_scale_trans)[kOutputHeightMax][kOutputWidthMax] = new int32_t[kOutputChannelsMax][kOutputHeightMax][kOutputWidthMax];

  T1 (*Pool_result)[kPoolOutputHeightMax][kPoolOutputWidthMax] = new T1[kOutputChannelsMax][kPoolOutputHeightMax][kPoolOutputWidthMax];
  // BatchNorm & Scale
  if(kBnEnable[layer]){
    for(int n_bn = 0; n_bn < N; n_bn++){
      for(int h_bn = 0; h_bn < H_Conv_OUT; h_bn++){
        for(int w_bn = 0; w_bn < W_Conv_OUT; w_bn++){
          // bn_result[n_bn][h_bn][w_bn] = bias_bn[bias_addr_offset + n_bn].alpha * conv_result[n_bn][h_bn][w_bn] + bias_bn[bias_addr_offset + n_bn].beta;
          bn_result[n_bn][h_bn][w_bn] = bias_bn[bias_addr_offset + n_bn].alpha * conv_result_scale_trans[n_bn][h_bn][w_bn] + bias_bn[bias_addr_offset + n_bn].beta;
        }
      }
    }
  }
  // ReLu
  if(kReluEnable[layer]){
    for(int n_rl = 0; n_rl < N; n_rl++){
      for(int h_rl = 0; h_rl < H_Conv_OUT; h_rl++){
        for(int w_rl = 0; w_rl < W_Conv_OUT; w_rl++){
          ReLu_result[n_rl][h_rl][w_rl] = (bn_result[n_rl][h_rl][w_rl] > 0) ? bn_result[n_rl][h_rl][w_rl] : 0;
          ReLu_result_scale_trans[n_rl][h_rl][w_rl] = ReLu_result[n_rl][h_rl][w_rl] * scale[layer][1];
          ReLu_result_scale_trans_pad[n_rl][h_rl+Pool_PAD][w_rl+Pool_PAD]=ReLu_result_scale_trans[n_rl][h_rl][w_rl];
          #ifdef PRINT_SIM_RELU_OUTPUT
            if(layer == PRINT_LAYER - 1)
            printf("SIM_RELU OUTPUT layer=%d n_rl=%d h_rl=%d w_rl=%d scale_1=%f relu_result=%f relu_result_scale_trans=%f relu_result_scale_trans_pad=%f\n", layer, n_rl, h_rl, w_rl, scale[layer][1], ReLu_result[n_rl][h_rl][w_rl], ReLu_result_scale_trans[n_rl][h_rl][w_rl], ReLu_result_scale_trans_pad[n_rl][h_rl+PAD][w_rl+PAD]);
          #endif
        }
      }
    }
  }

  // Pool operation
  if(kPoolEnable[layer] || kEndPoolEnable[layer]){
    for(int n_p = 0; n_p < N; n_p++){
        // for(int start_p_h = 0, h_p_r = 0; start_p_h < H_Pool_Out; h_p_r++, start_p_h = STRIDE_Pool * h_p_r){
        //   for(int start_p_w = 0, w_p_r = 0; start_p_w < W_Conv_OUT; w_p_r++, start_p_w = STRIDE_Pool * w_p_r){
        for(int start_p_h = 0, h_p_r = 0; (start_p_h+FPool)<= H_Pool_PAD; h_p_r++, start_p_h = STRIDE_Pool * h_p_r){
          for(int start_p_w = 0, w_p_r = 0; (start_p_w+FPool)<= W_Pool_PAD; w_p_r++, start_p_w = STRIDE_Pool * w_p_r){
            T1 max = 0.0;
            T1 average_pool_sum = 0.0;
            for(int h_p = start_p_h; (h_p < start_p_h + FPool) && (h_p < H_Pool_PAD); h_p++){
              for(int w_p = start_p_w; (w_p < start_p_w + FPool) && (w_p < W_Pool_PAD); w_p++){
                if(!kPoolType[layer]){
                  // if(ReLu_result[n_p][h_p][w_p] > max) max = ReLu_result[n_p][h_p][w_p];
                  if(ReLu_result_scale_trans_pad[n_p][h_p][w_p] > max) max = ReLu_result_scale_trans_pad[n_p][h_p][w_p];
                  #ifdef PRINT_SIM_POOL_MAX_OUTPUT
                    if(layer == PRINT_LAYER - 1)
                    printf("SIM_POOL OUTPUT layer=%d n_p=%d h_p=%d w_p=%d h_p_r=%d w_p_r=%d max=%f relu_result_scale_trans_pad=%f\n", layer, n_p, h_p, w_p, h_p_r, w_p_r, max, ReLu_result_scale_trans_pad[n_p][h_p][w_p]);
                  #endif
                }
                else{
                  average_pool_sum = + ReLu_result_scale_trans_pad[n_p][h_p][w_p];
                }
              }
            }
            if(!kPoolType[layer]){
              Pool_result[n_p][h_p_r][w_p_r] = max;
              #ifdef PRINT_SIM_POOL_MAX_OUTPUT
                    if(layer ==  PRINT_LAYER - 1)
                    printf("SIM_POOL OUTPUT layer=%d n_p=%d h_p_r=%d w_p_r=%d max=%f pool_result=%f\n", layer, n_p, h_p_r, w_p_r, max, Pool_result[n_p][h_p_r][w_p_r]);
              #endif
            }
            else{
              Pool_result[n_p][h_p_r][w_p_r] = average_pool_sum/(FPool*FPool);
            }
          }
        }
    }
  }

  // 3D Array to 1D
  // T1 *sim_output_middle = (T1*)malloc(sizeof(T1)*kOutputChannelsMax*kOutputHeightMax*kOutputWidthMax);
  // T1 *sim_output_middle = new T1[N*H_Conv_OUT*W_Conv_OUT];
  T1 *sim_output_middle = new T1[kOutputChannelsMax*kOutputHeightMax*kOutputWidthMax];
  for(int sim_n = 0; sim_n < N; sim_n++){
    // for(int sim_h_out = 0 ; sim_h_out < H_Conv_OUT; sim_h_out++){
    //   for(int sim_w_out = 0; sim_w_out < W_Conv_OUT; sim_w_out++){
    for(int sim_h_out = 0 ; sim_h_out < H_Pool_Out; sim_h_out++){
      for(int sim_w_out = 0; sim_w_out < W_Pool_Out; sim_w_out++){
        // sim_output_middle[sim_n * H_Conv_OUT * W_Conv_OUT + sim_h_out * W_Conv_OUT + sim_w_out] = (kPoolEnable[layer] || kEndPoolEnable[layer]) ? Pool_result[sim_n][sim_h_out][sim_w_out] : (kReluEnable[layer] ? ReLu_result[sim_n][sim_h_out][sim_w_out] : (kBnEnable[layer] ? bn_result[sim_n][sim_h_out][sim_w_out] : conv_result[sim_n][sim_h_out][sim_w_out]));
        sim_output_middle[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out] = (kPoolEnable[layer] || kEndPoolEnable[layer]) ? Pool_result[sim_n][sim_h_out][sim_w_out] : (kReluEnable[layer] ? ReLu_result_scale_trans[sim_n][sim_h_out][sim_w_out] : (kBnEnable[layer] ? bn_result[sim_n][sim_h_out][sim_w_out] : conv_result_scale_trans[sim_n][sim_h_out][sim_w_out]));
        if((layer == 0) || (layer == DEVICE_END_LAYER)){
          // sim_output = sim_output_middle;
          sim_output[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out] = sim_output_middle[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out];
        }
        else
        {
          sim_output[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out] = (T1)round(sim_output_middle[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out]);
        }

        #ifdef PRINT_SIM_OUTPUT
            if(layer == PRINT_LAYER - 1)
            printf("SIM_PE OUTPUT layer=%d sim_n=%d sim_h_out=%d sim_w_out=%d index=%d sim_output_middle=%f sim_output=%f\n", layer, sim_n, sim_h_out, sim_w_out, sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out, sim_output_middle[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out], sim_output[sim_n * H_Pool_Out * W_Pool_Out + sim_h_out * W_Pool_Out + sim_w_out]);
        #endif
      }
    }
  }

  delete(sim_output_middle);
  delete(Pool_result);
  delete(ReLu_result_scale_trans_pad);
  delete(ReLu_result_scale_trans);
  delete(ReLu_result);
  delete(bn_result);
  delete(conv_result_scale_trans);
  delete(conv_result_m);
  delete(conv_result);
  delete(input_raw_pad);
  // free(sim_output_middle);
  // delete(sim_output_middle);
}
