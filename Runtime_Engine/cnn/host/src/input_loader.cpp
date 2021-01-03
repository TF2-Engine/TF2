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

void LoadInputJpeg(char *image_name, float *input_raw, float *raw_images, int iter) {
  printf("load_input_image image_name=%s\n", image_name);

  std::vector<cv::Mat> channels;
  cv::Mat src, dst1, dst2;

  // unsigned char C = 3;
  // unsigned char H = 224;
  // unsigned char W = 224;

  unsigned char C = INPUT_IMAGE_C;
  unsigned char H = INPUT_IMAGE_H;
  unsigned char W = INPUT_IMAGE_W;
  
  int  H_middle = 256;
  int  W_middle = 256;

  // float googlenet_mean[3] = {104, 117, 123};
  float static_mean[3] = {0.485, 0.456, 0.406};
  float static_norm[3] = {0.229, 0.224, 0.225};

  FILE *fp;
  fp = fopen( "../host/model/mean.bin", "rb" );

  // 256 * 256 preprocess
  src = cv::imread( image_name );
  src.convertTo( dst1, CV_32FC3 );
  resize( dst1, dst2, cvSize( H_middle, W_middle ), 1 );
  //resize( src, dst2, cvSize( H_middle, W_middle ) );
  cv::split( dst2, channels ); 

  float *middle_images = (float*)malloc( sizeof(float) * C * H_middle * W_middle ); 

  int channel_order[3] = {2, 1, 0};

  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H_middle; h++) {
      for(int w = 0; w < W_middle; w++) {
        int addr = c * H_middle * W_middle + h * W_middle + w;
        float mean;
#if (defined RESNET50) || (defined RESNET50_PRUNED) || (defined SQUEEZENET)
        //fread( &mean, sizeof(float), 1, fp );
        mean = static_mean[c];
#else
        mean = googlenet_mean[c];
#endif
        //middle_images[addr] = channels[c].at<uchar>(h,w) - mean;
        middle_images[addr] = (channels[channel_order[c]].at<float>(h,w) / 255.0f - mean) / static_norm[c];
        //printf( "c=%d h=%d w=%d middle_images[%d]=%f mean=%f\n", c, h, w, addr, middle_images[addr], mean );
      }
    }
  }

  // crop operation
  int H_edge = ( H_middle - H ) / 2;
  int W_edge = ( W_middle - W ) / 2;
  
  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H; h++) {
      for(int w = 0; w < W; w++) {
        int addr1 = c * H_middle * W_middle + ( h + H_edge ) * W_middle + ( w + W_edge);
        int addr2 = c * H * W + h * W + w;
        raw_images[addr2] = middle_images[addr1];
        #ifdef FBIT_MASK
        input_raw[addr2] = raw_images[addr2];
        #endif
        //printf( "c=%d h=%d w=%d addr1=%d raw_images[%d]=%f\n", c, h, w, addr1, addr2, raw_images[addr2] );
        //printf( "c=%d h=%d w=%d images[%d]=%f\n", c, h, w, addr2, raw_images[addr2] );
      }
    }
  }

  #ifndef FBIT_MASK
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
        //printf("Input Raw i=%d j=%d k=%d input_raw=%f\n", i, j, k, input_raw[i * 114 * 114 + j * 114 + k]);
      }
    }
  }

  free(new_input);
  #endif

  fclose(fp);
  free(middle_images);
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
        //printf("Image c=%d h=%d w=%d raw_images=%f\n", c, h, w, raw_images[addr]);
        #ifdef FBIT_MASK
        input_raw[addr] = raw_images[addr];
        #endif
        #ifdef PRINT_H_INPUT_IMG_RAW
        printf("Image c=%d h=%d w=%d addr=%d raw_images=%f input_raw=%f\n", c, h, w, addr, raw_images[addr], input_raw[addr]);
        #endif
      }
    }
  }
  fclose(fp);

  #ifndef FBIT_MASK
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
        //printf("Input Raw i=%d j=%d k=%d input_raw=%f\n", i, j, k, input_raw[i * 114 * 114 + j * 114 + k]);
      }
    }
  }

  free(new_input);
  #endif
}

// input_raw[C][H][W]
// to
// input[CEIL(C, C_VECTOR)][H][CEIL(W, W_VECTOR)][W_VECTOR][C_VECTOR]
void InputConvert(float *input_raw, float *input, int num_images)
{
  // int C = kInputChannels[0];
  // int H = kInputHeight[0];
  // int W = kInputWidth[0];
  #ifndef FBIT_MASK
  int C = kInputChannels[DEVICE_START_LAYER];
  #else
  int C = kOutputChannels[DEVICE_START_LAYER-1];
  #endif
  int H = kInputHeight[DEVICE_START_LAYER];
  int W = kInputWidth[DEVICE_START_LAYER];

  for ( int n = 0; n < num_images; n++ ) {
    for (int cvec = 0; cvec < CEIL(C, C_VECTOR); cvec++) {
      for (int h = 0; h < H; h++) {
        for (int ww = 0; ww < CEIL(W, W_VECTOR); ww++) {
          for (int wvec = 0; wvec < W_VECTOR; wvec++) {
            for (int c = 0; c < C_VECTOR; c++) {
              #ifndef CHECK_H_SIMULATION_START_OUTPUT
              unsigned long long int addr = (unsigned long long int)
                                            n * CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            cvec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            h * CEIL(W, W_VECTOR) *NEXT_POWER_OF_2 (W_VECTOR * C_VECTOR) +
                                            ww * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            wvec * C_VECTOR +
                                            c;
              #endif
             int linear_c = cvec * C_VECTOR + c;
             int linear_w = ww * W_VECTOR + wvec;

             bool not_out_of_bounds = (linear_c < C && linear_w < W);
             unsigned long long int input_raw_addr = (unsigned long long int) linear_c * H * W + h * W + linear_w;
             #ifdef CHECK_H_SIMULATION_START_OUTPUT
             unsigned long long int addr = input_raw_addr;
             #endif
             input[addr] = not_out_of_bounds ? input_raw[input_raw_addr] : 0.0;
             #ifdef PRINT_H_INPUT_CONVERT
             printf("Input Convert cvec=%d h=%d ww=%d wvec=%d c=%d addr=%llu input_raw_addr=%llu input=%f input_raw=%f\n", cvec, h, ww, wvec, c, addr, input_raw_addr, input[addr], input_raw[input_raw_addr]);
             #endif
            }
          }
        }
      }
    }
  }
}

// input_raw[C][H][W]
// to
// input[CEIL(C, C_VECTOR)][H][CEIL(W, W_VECTOR)][W_VECTOR][C_VECTOR]
void OutputRConvert(real *output, float *output_pre, int num_images)
{
  int output_channel = kOutputChannels[DEVICE_END_LAYER - 1];
  int W = kPoolOutputWidth[DEVICE_END_LAYER - 1];
  int H = kPoolOutputHeight[DEVICE_END_LAYER - 1];

  int output_offset = OUTPUT_OFFSET + (num_images - 1) * OUTPUT_OFFSET;
  int concat_offset = kNStart[DEVICE_END_LAYER - 1] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);

  int ddr_write_offset = kDDRWriteBase[DEVICE_END_LAYER - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);

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
                    output_offset + concat_offset + ddr_write_offset +
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    ww * NARROW_N_VECTOR +
                    nn;
        //int addr_exp = ( n + kNStart[ NUM_LAYER - 1 ] ) * H * W + h * W + w; // for concat layer splitted branch pool output debug
        int addr_exp = n * H * W + h * W + w;

        output_pre[addr_exp] = output[addr_out] * 1.0;
        #ifdef PRINT_H_OUTPUT_CONVERT
        printf("Output Convert output_pre=%.6f output=%.6f addr_exp=%d addr_out=%d output_offset=%d concat_offset=%d ddr_write_offset=%d n=%d h=%d w=%d\n", output_pre[addr_exp],1.0*output[addr_out], addr_exp, addr_out, output_offset, concat_offset, ddr_write_offset, n, h, w);
        #endif
      }
    }
  }

}

