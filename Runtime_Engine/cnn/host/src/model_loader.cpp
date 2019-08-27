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
/*
#define DIM 224
#define PAD_H_0 3
#define PAD_W_0 3
#define NDIM_H (DIM+2*PAD_H_0)
#define NDIM_W (DIM+2*PAD_W_0)
#define OUTPUT_WIDTH  (NDIM_W/2)
#define OUTPUT_HEIGHT (NDIM_H/2)
#define OUTPUTDIMS  114  ///114-3+1=112;

#define MAX_OUTPUT_CHANNEL 2048

void feature_trans(float *Feas,float *finals_feas)
{
  float Trans_Pad[NDIM_H][NDIM_W] = {{0.0}};
  for(int i = 0; i < DIM; i++) {
    for(int j = 0; j < DIM; j++) {
      Trans_Pad[i + PAD_H_0][j + PAD_W_0] = Feas[i * DIM + j];
    }
  }	

  float Fea_media[3][NDIM_H][NDIM_W] = {{{0.0}}};
  for(int i = 0; i < NDIM_H; i++) {
    for(int j = 0; j < NDIM_W; j++) {
      Fea_media[j % 2][i][j / 2] = Trans_Pad[i][j];
    }
  }

  for(int i = 0; i< NDIM_H; i++) {
    for(int j = 0; j < NDIM_W - 1; j++) {
      Fea_media[2][i][j] = Fea_media[0][i][j + 1];
    }
  }

  float height_trans[6][NDIM_H / 2][NDIM_W / 2] = {{{0}}};
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < NDIM_H; j++) {
      for(int k = 0; k < NDIM_W / 2; k++) {
        height_trans[i * 2 + j % 2][j / 2][k] = Fea_media[i][j][k];
      }
    }
  }
        	 
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < NDIM_H / 2; j++) {
      for(int k = 0; k < NDIM_H / 2; k++) {
        finals_feas[i * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k]=height_trans[i][j][k];
      }
    }	
  }

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < NDIM_H / 2; j++) {
      for(int k = 0; k < NDIM_W / 2; k++) {
        finals_feas[(i + 6) * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k] = height_trans[i * 2][j + 1][k];
      }
    }
  }	
}
*/
#define CHAS 1
#define FILTER_FIRST 7
#define R_A_D 3

void filter_trans(real *filter_input, real *Transform_input)
{
  real filter_media[CHAS * 2][FILTER_FIRST][FILTER_FIRST];

  for(int i = 0; i < CHAS * 2; i++) {
    for(int j = 0; j < FILTER_FIRST; j++) {
      for(int k = 0; k < FILTER_FIRST; k++) {
        filter_media[i][j][k] = 0x40;
      }
    }
  }

  real filter_width[CHAS * 3][FILTER_FIRST][FILTER_FIRST];

  // order is channel, heights, width
  for(int i = 0; i < CHAS * 3; i++) {
    for(int j = 0; j < FILTER_FIRST; j++) {
      for(int k = 0; k < FILTER_FIRST; k++) {
        filter_width[i][j][k] = 0x40;
      }
    }
  }

  for(int i = 0; i < FILTER_FIRST; i++) {
    for(int j = 0; j < FILTER_FIRST; j++) {
      filter_media[j % 2][i][j / 2] = filter_input[i * FILTER_FIRST + j];
    }
  }

  for(int t = 0; t < 2; t++) {
    for(int i = 0; i < FILTER_FIRST; i++) {
      for(int j = 0; j < FILTER_FIRST / 2; j++) {
        filter_width[t][i][j] = filter_media[t][i][j];
      }
    }
  }

  for(int i = 0; i < FILTER_FIRST; i++) {
    filter_width[2][i][2] = filter_media[0][i][3];
  }

  real filter_heights[6*CHAS][FILTER_FIRST][FILTER_FIRST];

  for(int i = 0; i < CHAS * 6; i++) {
    for(int j = 0; j < FILTER_FIRST; j++) {
      for(int k = 0; k < FILTER_FIRST; k++) {
        filter_heights[i][j][k] = 0;
      }
    }
  }

  for(int i = 0; i < 3 * CHAS; i++) {
    for(int j = 0; j < FILTER_FIRST / 2; j++) {
      for(int k = 0; k < FILTER_FIRST; k++) {
        filter_heights[i * 2 + k % 2][k / 2][j] = filter_width[i][k][j];
      }
    }
  }

  for(int i = 0; i < 6 * CHAS; i++) {
    for(int j = 0; j < FILTER_FIRST / 2; j++) {
      for(int k = 0; k < FILTER_FIRST / 2; k++) {
        Transform_input[i * R_A_D * R_A_D + j * R_A_D + k] = filter_heights[i][j][k];
      }
    }

    if(i % 2 == 0) {
      for(int k = 0; k < FILTER_FIRST / 2; k++) {  ///0,1,2分别???,7,8
        Transform_input[(i / 2 + 6) * R_A_D * R_A_D + 2 * R_A_D + k] = filter_heights[i][3][k];
      }
    }
  }
}

char Get_real(float data, char expand){
  bool sign = 0;
  char vals = 0;
  if(fabs(data) < 1.0e-05)
    return 0x40; // 0x40 represents 0
  if(data < 0){
    sign = 1;
    data =- data;
  }
	
  for(int i = 0; i < 15; i++){
    float temps = 1.0f / (1 << i);
    if(data > 0.99 * temps && data < 1.01 * temps){
      vals = i;
      break;
    }
  }

  char oups = expand - vals; // to minus vals because vals is the N of 2^-N
  if(oups < 0) {
    oups = 0;
  }

  if(sign) {
    oups = oups | 0x80; //0000,0000  0100,0000==0, 0010,??ͷ?ı?ʾ??????
  }

  return oups;
}

// filter_raw[N][C][H][W]
void load_model(char *filename, real *filter_raw, bias_bn_param_t *bias_bn, char *q) {
  FILE* infile;
  if((infile=fopen(filename,"rb"))==NULL){
    printf("Error in search of %s in load_model modulus\n",filename);
    exit(1);
  }

  int filter_addr_offset = 0;
  int bias_addr_offset = 0;
  
  // load whole model
  int q_offset = 0;
  for(int c_index = 0; c_index < NUM_LAYER; c_index++) {
    float mean[MAX_OUTPUT_CHANNEL] = {0};
    float variance[MAX_OUTPUT_CHANNEL] = {0};
    float scale_factor = 0;
    float alpha[MAX_OUTPUT_CHANNEL] = {0};
    float beta[MAX_OUTPUT_CHANNEL] = {0};
    
    int C = c_index == 0 ? INPUT_IMAGE_C : input_channels[c_index];
      
    int H = c_index == 0 ? 7 : filter_size[c_index];
    int W = c_index == 0 ? 7 : filter_size[c_index];
    int END_N = output_channels[c_index];
    	  
    if( !ipool_enable[c_index] ) { 
      for(int n = 0; n < END_N; n++) {
        for(int c = 0; c < C; c++) {
          int q_fixed;
            
          q_fixed = q[ input_layer[c_index] * MAX_OUT_CHANNEL + c ];
          
          int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n]; 
          char expand = INFLAT + q_fixed - q_fixed_gap;
          
          for(int h = 0; h < H; h++) {
            for(int w = 0; w < W; w++) {
              float filter_tems;
              fread( &filter_tems, sizeof(float), 1, infile );
              filter_raw[filter_addr_offset + n * C * H * W + c * H * W + h * W + w] = Get_real(filter_tems, expand);
            }
          }
        }
      }
    }

    // bias
    if( bias_enable[c_index] ) {				  
      for(int n = 0; n < END_N; n++) {
        int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n];
        float bias_trans_coe = 1 << (INFLAT - q_fixed_gap);
	float bias_data;
        fread( &bias_data, sizeof(float), 1, infile );
	bias_bn[bias_addr_offset + n].bias = bias_data * bias_trans_coe;    
      }
    } else {
      for(int n = 0; n < END_N; n++) {
        bias_bn[bias_addr_offset + n].bias=0;
      }
    }	

    if( bn_enable[c_index] ) {
      // mean
      for(int n = 0; n < END_N; n++) {
        fread( &mean[n], sizeof(float), 1, infile );
      }

      // variance
      for(int n = 0; n < END_N; n++) {
        fread( &variance[n], sizeof(float), 1, infile );
      }

      // scale_factor
      fread( &scale_factor, sizeof(float), 1, infile );

      // alpha
      for(int n = 0; n < END_N; n++) {
        fread( &alpha[n], sizeof(float), 1, infile );
      }

      // beta
      for(int n = 0; n < END_N; n++) {
        fread( &beta[n], sizeof(float), 1, infile );
      }
    }

    // alpha data
    for(int n = 0; n < END_N; n++) {
      float a;
      float b;
      float alpha_data;
      float beta_data;
      float eps = 0.00001;
      
      a = mean[n] / scale_factor;
      b = sqrt( variance[n] / scale_factor + eps );
      alpha_data = bn_enable[c_index] ? alpha[n] / b : 1.0;
      beta_data = bn_enable[c_index] ? -( alpha[n] / b * a ) + beta[n] : 0.0;
      
      int q_fixed_gap = q[q_offset + MAX_OUT_CHANNEL + n];
      float bias_trans_coe = 1 << (INFLAT - q_fixed_gap);
      bias_bn[bias_addr_offset + n].alpha = alpha_data * pow(2, ALPHA_INFLAT);
      bias_bn[bias_addr_offset + n].beta = beta_data > 0 ? (bias_trans_coe * beta_data + 0.5) : (bias_trans_coe * beta_data - 0.5);
    }

    if( c_index < ( NUM_LAYER - 1 ) ) {
      filter_addr_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR);
      bias_addr_offset += MAX_BIAS_SIZE;
    }

    q_offset += MAX_OUT_CHANNEL;
  }

  fclose(infile);
  
  // convert first layer kernel size to 3 
  int Trans_size = 64 * 3 * 3 * 3 * 3 * 3;
  real *first_layer_filter_trans = (real*)malloc(sizeof(real) * Trans_size);
  memset(first_layer_filter_trans, 0, sizeof(real)*64*9*27);
  for(int out_channel = 0; out_channel < output_channels[0]; out_channel++){
    for(int input_channel = 0; input_channel < 3; input_channel++){
      filter_trans(filter_raw + input_channel * 49 + out_channel * 49 * 3, first_layer_filter_trans + 9 * 9 * 3 * out_channel + 3 * 3 * 3 * 3 * input_channel);
    }
  }
 
  for(int i = 0; i < Trans_size; i++)
    filter_raw[i] = first_layer_filter_trans[i];
 
  free(first_layer_filter_trans);
}

// filter_raw[ NUM_CONVOLUTIONS ][ K ][ C ][ R ][ S ]
// to
// filter[ NUM_CONVOLUTIONS ][ MAX_FILTER_SIZE ], i.e. filter_buffer[c_index][dl][kvec][cvec][r][ss][k][svec][c]
void filter_convert(real *filter, real *filter_raw, real *filter_real) {
  unsigned long long int conv_filter_offset = 0;
  unsigned long long int conv_filter_raw_offset = 0;

  for(int c_index = 0; c_index < NUM_LAYER; c_index++) {
    int C = input_channels[c_index];

    // copy filter      
    int R = filter_size[c_index];
    int S = filter_size[c_index];
    int END_SS = CEIL(S, S_VECTOR);
    int K = output_channels[c_index];
    int END_CVEC = R == 1 ? CEIL(C, C_VECTOR * S_VECTOR) : CEIL(C, C_VECTOR);
    int END_KVEC = CEIL(K, K_VECTOR);

    for(int kvec = 0; kvec < END_KVEC; kvec++) {
      for(int cvec = 0; cvec < END_CVEC; cvec++) {
        for(int r = 0; r < R; r++) {
          for(int ss = 0; ss < END_SS; ss++) {
            for(int k = 0; k < K_VECTOR; k++) {

              //Following is for the storage os SVEC*C_VECTOR data in filter_buf;
              real filter_buf[S_VECTOR][C_VECTOR] = {{0}};//S_VECTOR*C_VECTOR 
              //for(int svec = 0; svec < SVEC; svec++) { 
              for(int svec = 0; svec < S_VECTOR; svec++) {
                for(int c = 0; c < C_VECTOR; c++) {
                  int linear_k = kvec * K_VECTOR + k;
                  int linear_c = R == 1 ? cvec * C_VECTOR * S_VECTOR + C_VECTOR * svec + c : cvec * C_VECTOR + c;
                  int linear_s = R == 1 ? 0 : ss * S_VECTOR + svec;
                  bool not_out_of_bounds = ( linear_k < K && linear_c < C && linear_s < S );
                  
                  unsigned long long int filter_raw_addr = conv_filter_raw_offset + linear_k * (C * R * S) + linear_c * R * S + r * S + linear_s;
 
                  unsigned long long int addr =
                      conv_filter_offset +
                      kvec * END_CVEC * R * END_SS * K_VECTOR * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR) +
                      cvec * R * END_SS * K_VECTOR * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR) +
                      r * END_SS * K_VECTOR * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR) +
                      ss * K_VECTOR * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR) +
                      k * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR) +
                      svec * C_VECTOR +
                      c;

                  if(not_out_of_bounds) {
                    filter_real[addr] = filter_raw[filter_raw_addr];
                  }
                }
              }
            }
          }
        }
      }
    }
    if( c_index < ( NUM_LAYER - 1 ) ) {
      conv_filter_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR);
      conv_filter_raw_offset += MAX_FILTER_SIZE * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR);
    }
  }
}


