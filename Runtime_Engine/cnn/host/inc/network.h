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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "includes.h"

// Image loading thread args
struct img_load_args_t {
  char *path;
  int size;
  int raw_size;
  int num_images;
};

/*
static int wait_after_conv[NUM_CONVOLUTIONS] = {
  0,
  0,
  0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0,    0,    0,    0,    0,    0,    0,
  0
};
*/
class NetWork {
public:
  NetWork();
  bool init(OpenCLFPGA &platform, char *model_file, char* q_file, char *image_file, int num_images);
  bool init_network();
  bool init_buffer();
  void cleanup();

  OpenCLFPGA platform;
  char *model_file;
  char *q_file;
  char *image_file;
  int num_images;
  float* input_raw = NULL;
  float* input = NULL;
  real* input_real=NULL;
  //real* filter_raw = NULL;
  //real* filter = NULL;
  //real* filter_real = NULL;
  char* q = NULL;
  //bias_bn_param_t *bias_bn = NULL;
  real* output = NULL;
  int top_labels[5];

  cl_mem input_buffer[2];
  cl_mem feature_ddr;

  cl_mem filter_buffer;
  cl_mem bias_bn_buffer;
  cl_mem wait_after_conv_cycles;

private:
  //float* input = NULL;
  real* filter_raw = NULL;
  real* filter = NULL;
  real* filter_real = NULL;
  bias_bn_param_t *bias_bn = NULL;
};

#endif
