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

#include "model_loader.h"

#define MAX_OUTPUT_CHANNEL 2048

#define CHAS 1
#define FILTER_FIRST 7
#define R_A_D 3

// Warning: This function will be deprecated in future version.
void filter_trans(real *filter_input, real *Transform_input) {
  real filter_media[CHAS * 2][FILTER_FIRST][FILTER_FIRST];

  for (int i = 0; i < CHAS * 2; i++) {
    for (int j = 0; j < FILTER_FIRST; j++) {
      for (int k = 0; k < FILTER_FIRST; k++) {
        filter_media[i][j][k] = 0x40;
      }
    }
  }

  real filter_width[CHAS * 3][FILTER_FIRST][FILTER_FIRST];

  // order is channel, heights, width
  for (int i = 0; i < CHAS * 3; i++) {
    for (int j = 0; j < FILTER_FIRST; j++) {
      for (int k = 0; k < FILTER_FIRST; k++) {
        filter_width[i][j][k] = 0x40;
      }
    }
  }

  for (int i = 0; i < FILTER_FIRST; i++) {
    for (int j = 0; j < FILTER_FIRST; j++) {
      filter_media[j % 2][i][j / 2] = filter_input[i * FILTER_FIRST + j];
    }
  }

  for (int t = 0; t < 2; t++) {
    for (int i = 0; i < FILTER_FIRST; i++) {
      for (int j = 0; j < FILTER_FIRST / 2; j++) {
        filter_width[t][i][j] = filter_media[t][i][j];
      }
    }
  }

  for (int i = 0; i < FILTER_FIRST; i++) {
    filter_width[2][i][2] = filter_media[0][i][3];
  }

  real filter_heights[6*CHAS][FILTER_FIRST][FILTER_FIRST];

  for (int i = 0; i < CHAS * 6; i++) {
    for (int j = 0; j < FILTER_FIRST; j++) {
      for (int k = 0; k < FILTER_FIRST; k++) {
        filter_heights[i][j][k] = 0;
      }
    }
  }

  for (int i = 0; i < 3 * CHAS; i++) {
    for (int j = 0; j < FILTER_FIRST / 2; j++) {
      for (int k = 0; k < FILTER_FIRST; k++) {
        filter_heights[i * 2 + k % 2][k / 2][j] = filter_width[i][k][j];
      }
    }
  }

  for (int i = 0; i < 6 * CHAS; i++) {
    for (int j = 0; j < FILTER_FIRST / 2; j++) {
      for (int k = 0; k < FILTER_FIRST / 2; k++) {
        Transform_input[i * R_A_D * R_A_D + j * R_A_D + k] = filter_heights[i][j][k];
      }
    }

    if (i % 2 == 0) {
      for (int k = 0; k < FILTER_FIRST / 2; k++) {
        Transform_input[(i / 2 + 6) * R_A_D * R_A_D + 2 * R_A_D + k] = filter_heights[i][3][k];
      }
    }
  }
}

char Get_real(float data, char expand) {
  bool sign = 0;
  char vals = 0;
  if (fabs(data) < 1.0e-05)
    return 0x40; // 0x40 represents 0
  if (data < 0) {
    sign = 1;
    data =- data;
  }
	
  for (int i = 0; i < 15; i++) {
    float temps = 1.0f / (1 << i);
    if (data > 0.99 * temps && data < 1.01 * temps) {
      vals = i;
      break;
    }
  }

  char oups = expand - vals; // to minus vals because vals is the N of 2^-N
  if (oups < 0) {
    oups = 0;
  }

  if (sign) {
    oups = oups | 0x80;
  }

  return oups;
}

// filter_raw[N][C][H][W]
void LoadModel(char *filename, real *filter_raw, BiasBnParam *bias_bn, char *q) {
  FILE* infile;
  if ((infile = fopen(filename, "rb")) == NULL) {
    printf("Error in search of %s in LoadModel modulus\n",filename);
    exit(1);
  }

  int filter_addr_offset = 0;
  int bias_addr_offset = 0;
  
  // load whole model
  int q_offset = 0;
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    float mean[MAX_OUTPUT_CHANNEL] = {0};
    float variance[MAX_OUTPUT_CHANNEL] = {0};
    float scale_factor = 0;
    float alpha[MAX_OUTPUT_CHANNEL] = {0};
    float beta[MAX_OUTPUT_CHANNEL] = {0};
    
    int C = layer == 0 ? INPUT_IMAGE_C : kInputChannels[layer];
      
    int H = layer == 0 ? FIRST_FILTER_SIZE : kFilterSize[layer];
    int W = layer == 0 ? FIRST_FILTER_SIZE : kFilterSize[layer];
    int N = kOutputChannels[layer];
    if (!kIpoolEnable[layer]) { 
      for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
          int q_fixed;
            
          q_fixed = q[kInputLayer[layer] * MAX_OUT_CHANNEL + c];
          
          int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n];
          char expand = INFLAT + q_fixed - q_fixed_gap;
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              float filter_tems;

	      if (getNetwork() == kYandi) {
		// For yandi network, conv layer23 is specifal.
		// channel 0-23 has no filter data.
	        if (layer == 23 && n < 24) {
                  filter_raw[filter_addr_offset + n * C * H * W + c * H * W + h * W + w] = 0x40; // Zero.
	        } else {
                  fread(&filter_tems, sizeof(float), 1, infile);
                  filter_raw[filter_addr_offset + n * C * H * W + c * H * W + h * W + w] = Get_real(filter_tems, expand);
	        }
	      } else {
                fread(&filter_tems, sizeof(float), 1, infile);
                filter_raw[filter_addr_offset + n * C * H * W + c * H * W + h * W + w] = Get_real(filter_tems, expand);
	      }
            }
          }
        }
      }
    }

    // bias
    if (kBiasEnable[layer]) {				  
      for (int n = 0; n < N; n++) {
        int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n];
        float bias_trans_coe = 1 << (INFLAT - q_fixed_gap);
	float bias_data;

	if (getNetwork() == kYandi) {
	  if (layer == 23 && n < 24) {
	    bias_bn[bias_addr_offset + n].bias = 0;
	  } else {
            fread( &bias_data, sizeof(float), 1, infile );
	    bias_bn[bias_addr_offset + n].bias = bias_data * bias_trans_coe;
	  }
	} else {
          fread( &bias_data, sizeof(float), 1, infile );
	  bias_bn[bias_addr_offset + n].bias = bias_data * bias_trans_coe;
	}
      }
    } else {
      for (int n = 0; n < N; n++) {
        bias_bn[bias_addr_offset + n].bias=0;
      }
    }	

    if (kBnEnable[layer]) {
      // mean
      for (int n = 0; n < N; n++) {
        fread(&mean[n], sizeof(float), 1, infile);
      }

      // variance
      for (int n = 0; n < N; n++) {
        fread( &variance[n], sizeof(float), 1, infile);
      }

      // scale_factor
      fread(&scale_factor, sizeof(float), 1, infile);

      // alpha
      for (int n = 0; n < N; n++) {
        fread(&alpha[n], sizeof(float), 1, infile);
      }

      // beta
      for (int n = 0; n < N; n++) {
        fread(&beta[n], sizeof(float), 1, infile);
      }
    }

    // alpha data
    for (int n = 0; n < N; n++) {
      float a;
      float b;
      float alpha_data;
      float beta_data;
      float eps = 0.00001;
      
      a = mean[n] / scale_factor;
      b = sqrt( variance[n] / scale_factor + eps );
      alpha_data = kBnEnable[layer] ? alpha[n] / b : 1.0;
      beta_data = kBnEnable[layer] ? - (alpha[n] / b * a) + beta[n] : 0.0;
      
      int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n];
      float bias_trans_coe = 1 << (INFLAT - q_fixed_gap);
      bias_bn[bias_addr_offset + n].alpha = alpha_data * pow(2, ALPHA_INFLAT);
      bias_bn[bias_addr_offset + n].beta = beta_data > 0 ? (bias_trans_coe * beta_data + 0.5) : (bias_trans_coe * beta_data - 0.5);
    }

    if (layer < ( NUM_LAYER - 1)) {
      filter_addr_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
      bias_addr_offset += MAX_BIAS_SIZE;
    }

    q_offset += MAX_OUT_CHANNEL;
  }

  fclose(infile);
  
// For googlenet or resnet50, convert first layer kernel size to 3
  if (getNetwork() == kGooglenet || getNetwork() == kResnet50) {
    int Trans_size = 64 * 3 * 3 * 3 * 3 * 3;
    real *first_layer_filter_trans = (real*)malloc(sizeof(real) * Trans_size);
    memset(first_layer_filter_trans, 0, sizeof(real)*64*9*27);
    for (int out_channel = 0; out_channel < kOutputChannels[0]; out_channel++) {
      for (int input_channel = 0; input_channel < 3; input_channel++) {
        filter_trans(filter_raw + input_channel * 49 + out_channel * 49 * 3, first_layer_filter_trans + 9 * 9 * 3 * out_channel + 3 * 3 * 3 * 3 * input_channel);
      }
    }
    for (int i = 0; i < Trans_size; i++)
      filter_raw[i] = first_layer_filter_trans[i];
    free(first_layer_filter_trans);
    }
}
// filter_raw[NUM_CONVOLUTIONS][N][C][FH][FW]
// to
// filter[NUM_CONVOLUTIONS][MAX_FILTER_SIZE], i.e. filter_buffer[layer][n_vec][c_vec][fh_vec][fw_vec][n][fw_inc][c_inc]
void FilterConvert(real *filter, real *filter_raw, real *filter_real) {
  unsigned long long int conv_filter_offset = 0;
  unsigned long long int conv_filter_raw_offset = 0;

  for (int layer = 0; layer < NUM_LAYER; layer++) {
    int C = kInputChannels[layer];

    // copy filter      
    int FH = kFilterSize[layer];
    int FW = kFilterSize[layer];
    int FW_VEC = CEIL(FW, FW_VECTOR);
    int N = kOutputChannels[layer];
    int C_VEC = FH == 1 ? CEIL(C, C_VECTOR * FW_VECTOR) : CEIL(C, C_VECTOR);
    int N_VEC = CEIL(N, N_VECTOR);

    for (int n_vec = 0; n_vec < N_VEC; n_vec++) {
      for (int c_vec = 0; c_vec < C_VEC; c_vec++) {
        for (int fh = 0; fh < FH; fh++) {
          for (int fw_vec = 0; fw_vec < FW_VEC; fw_vec++) {
            for (int n_inc = 0; n_inc < N_VECTOR; n_inc++) {

              // Following is for the storage of FW_VECTOR*C_VECTOR data in filter_buf;
              real filter_buf[FW_VECTOR][C_VECTOR] = {{0}};//FW_VECTOR*C_VECTOR 
              // for(int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) { 
              for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
                for (int c_inc = 0; c_inc < C_VECTOR; c_inc++) {
                  int n = n_vec * N_VECTOR + n_inc;
                  int c = FH == 1 ? c_vec * C_VECTOR * FW_VECTOR + C_VECTOR * fw_inc + c_inc : c_vec * C_VECTOR + c_inc;
                  int fw = FH == 1 ? 0 : fw_vec * FW_VECTOR + fw_inc;
                  bool not_out_of_bounds = ( n < N && c < C && fw < FW );
                  
                  unsigned long long int filter_raw_addr = conv_filter_raw_offset + n * (C * FH * FW) + c * FH * FW + fh * FW + fw;
 
                  unsigned long long int addr =
                      conv_filter_offset +
                      n_vec * C_VEC * FH * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                      c_vec * FH * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                      fh * FW_VEC * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                      fw_vec * N_VECTOR * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                      n_inc * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR) +
                      fw_inc * C_VECTOR +
                      c_inc;

                  if (not_out_of_bounds) {
                    filter_real[addr] = filter_raw[filter_raw_addr];
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (layer < (NUM_LAYER - 1)) {
      conv_filter_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
      conv_filter_raw_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
    }
  }
}


