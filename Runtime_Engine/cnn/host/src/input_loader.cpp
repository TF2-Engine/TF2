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

#include "input_loader.h"

#define DIM 224
#define PAD_H_0 3
#define PAD_W_0 3
#define NDIM_H (DIM+2*PAD_H_0)
#define NDIM_W (DIM+2*PAD_W_0)
#define OUTPUT_WIDTH (NDIM_W/2)
#define OUTPUT_HEIGHT (NDIM_H/2)
#define OUTPUTDIMS 114  // 114-3+1=112;

void feature_trans(float *Feas,float *finals_feas)
{
  float Trans_Pad[NDIM_H][NDIM_W] = {{0.0}};
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      Trans_Pad[i + PAD_H_0][j + PAD_W_0] = Feas[i * DIM + j];
    }
  }

  float Fea_media[3][NDIM_H][NDIM_W] = {{{0.0}}};
  for (int i = 0; i < NDIM_H; i++) {
    for (int j = 0; j < NDIM_W; j++) {
      Fea_media[j % 2][i][j / 2] = Trans_Pad[i][j];
    }
  }

  for (int i = 0; i< NDIM_H; i++) {
    for (int j = 0; j < NDIM_W - 1; j++) {
      Fea_media[2][i][j] = Fea_media[0][i][j + 1];
    }
  }

  float height_trans[6][NDIM_H / 2][NDIM_W / 2] = {{{0}}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < NDIM_H; j++) {
      for (int k = 0; k < NDIM_W / 2; k++) {
        height_trans[i * 2 + j % 2][j / 2][k] = Fea_media[i][j][k];
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < NDIM_H / 2; j++) {
      for (int k = 0; k < NDIM_H / 2; k++) {
        finals_feas[i * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k]=height_trans[i][j][k];
      }
    }
 }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < NDIM_H / 2; j++) {
      for (int k = 0; k < NDIM_W / 2; k++) {
        finals_feas[(i + 6) * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k] = height_trans[i * 2][j + 1][k];
      }
    }
  }
}

// load input raw image data:[C][H][W]
void LoadInputImage(char *image_name, float *input_raw, float *raw_images, int iter) {
  INFO("LoadInputImage image_name=%s\n", image_name);
  unsigned char C = INPUT_IMAGE_C;
  unsigned char H = INPUT_IMAGE_H;
  unsigned char W = INPUT_IMAGE_W;

  FILE *fp;
  if ((fp = fopen(image_name, "rb"))==NULL){
    printf("load input image : %s Error\n",image_name);
    exit(1);
  }

  for (int c = 0; c < C; c++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int addr = c * H * W + h * W + w;
        fread( &raw_images[addr], sizeof(float), 1, fp );
      }
    }
  }
  fclose(fp);

  // The below code is just for Resnet50 and GoogLeNet and will remove later.  
  unsigned char CC = 27;
  unsigned char HH = 115;
  unsigned char WW = 115;
  float *new_input=(float*)malloc(sizeof(float) * CC * HH * WW + 1000);
  for (int i = 0; i < 3; i++) {
    feature_trans(raw_images + i * H * W, new_input + 9 * i * OUTPUT_HEIGHT * OUTPUT_WIDTH);
  }

  for (int i = 0; i < 27; i++) {
    for (int j = 0; j < 114; j++) {
      for (int k = 0; k < 114; k++) {
        input_raw[i * 114 * 114 + j * 114 + k]=new_input[i * WW * HH + j * WW + k];
      }
    }
  }

  free(new_input);
}

// input_raw[C][H][W]
// to
// input[CEIL(C, C_VECTOR)][H][CEIL(W, W_VECTOR)][W_VECTOR][C_VECTOR]
void InputConvert(float *input_raw, float *input, int num_images)
{
  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];

  for ( int n = 0; n < num_images; n++ ) {
    for (int cvec = 0; cvec < CEIL(C, C_VECTOR); cvec++) {
      for (int h = 0; h < H; h++) {
        for (int ww = 0; ww < CEIL(W, W_VECTOR); ww++) {
          for (int wvec = 0; wvec < W_VECTOR; wvec++) {
            for (int c = 0; c < C_VECTOR; c++) {
              unsigned long long int addr = (unsigned long long int)
                                            n * CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            cvec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            h * CEIL(W, W_VECTOR) *NEXT_POWER_OF_2 (W_VECTOR * C_VECTOR) +
                                            ww * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            wvec * C_VECTOR +
                                            c;

              int linear_c = cvec * C_VECTOR + c;
              int linear_w = ww * W_VECTOR + wvec;

              bool not_out_of_bounds = (linear_c < C && linear_w < W);
              unsigned long long int input_raw_addr = (unsigned long long int) linear_c * H * W + h * W + linear_w;

              input[addr] = not_out_of_bounds ? input_raw[input_raw_addr] : 0.0;
              //printf("input[%d]=%f\n", addr, input[addr]);
            }
          }
        }
      }
    }
  }
}

