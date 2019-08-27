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

#ifndef __CNN_GOOGLENET__
#define __CNN_GOOGLENET__

//-----------------------------------------------------------------------
//                                                                     
// CNN Network Ralated Paramters     
//                                                                     
//-----------------------------------------------------------------------

// number of layer
//#define NUM_LAYER 67
#define NUM_LAYER 67

// number of convolution layers
#define NUM_CONVOLUTIONS 67
#define NUM_Q_LAYERS ( NUM_CONVOLUTIONS + 1 + 9 ) // 1 is for input data quantization value, 9 is for concatenate layer, their qs are stored seperately.

#define INPUT_IMAGE_C 3
#define INPUT_IMAGE_H 224
#define INPUT_IMAGE_W 224

#define MAX_OUT_CHANNEL 1024
#define MAX_POOL_OUTPUT_WVEC CEIL(56, W_VECTOR)

// set size of feature buffer
//#define DDR_PAGE_SIZE0 (CEIL(480, C_VECTOR) * 56 * CEIL(56, W_VECTOR)) //0/2/4/6... layer Abraham NOT SURE
#define DDR_PAGE_SIZE0 (CEIL(480, C_VECTOR) * 56 * CEIL(56, W_VECTOR))
#define DDR_PAGE_SIZE1 (CEIL(256, C_VECTOR) * 56 * CEIL(56, W_VECTOR))  //1/3/5... layer Abraham NOT SURE
#define DDR_SIZE (DDR_PAGE_SIZE0 + DDR_PAGE_SIZE1)
//#define DDR_SIZE (DDR_SLICE_SIZE * 2)

#define CACHE_PAGE_SIZE (CEIL(27, C_VECTOR) * 114 * CEIL(114, W_VECTOR)) // single buffer size: be calculated from the layer of max slice size
#define CACHE_SIZE (CACHE_PAGE_SIZE * 3)

// set size of filter buffer
#define FILTER_CACHE_PAGE_SIZE1 (NEXT_DIVISIBLE(192, C_VECTOR) * 3 * CEIL(3, S_VECTOR))
#define FILTER_CACHE_PAGE_SIZE2 (NEXT_DIVISIBLE(1024, C_VECTOR * S_VECTOR))
#define FILTER_CACHE_PAGE_SIZE  (MYMAX2(FILTER_CACHE_PAGE_SIZE1, FILTER_CACHE_PAGE_SIZE2))

// Set size of host filter and bias buffer of each layer.
#define MAX_FILTER_SIZE1 (NEXT_POWER_OF_2(CEIL(2048, C_VECTOR) * 1 * CEIL(1, S_VECTOR) * NEXT_DIVISIBLE(2048, K_VECTOR) * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE2 (NEXT_POWER_OF_2(CEIL(1024, C_VECTOR) * 3 * CEIL(3, S_VECTOR) * NEXT_DIVISIBLE(1024, K_VECTOR) * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE_TEMP ((MAX_FILTER_SIZE1 > MAX_FILTER_SIZE2) ? MAX_FILTER_SIZE1 : MAX_FILTER_SIZE2)
#define MAX_FILTER_SIZE (CEIL(MAX_FILTER_SIZE_TEMP, NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))

#define MAX_BIAS_SIZE  NEXT_DIVISIBLE(2048, K_VECTOR)

#define DDR_BLOCK_SIZE DDR_PAGE_SIZE0
#define D0 0
#define D1 0
#define D2 DDR_BLOCK_SIZE

#define OUTPUT_OFFSET (3 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR))
//#define OUTPUT_SIZE (CEIL(CEIL(1000, C_VECTOR) * 1 * CEIL(1, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR) * NEXT_POWER_OF_2(C_VECTOR), NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR)) * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR))
// -------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
// Convolution Parametres of each layer
//-------------------------------------------------------------------------------------------------------------------
CONSTANT int ddr_write_base[NUM_CONVOLUTIONS] = {
  D0,         
  D0,         
  D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0, D0, D0, D0, D0, D0, D0, 
  D0
};

CONSTANT int filter_size[NUM_CONVOLUTIONS] =
{
  3, 
  1, 
  3, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1, 1, 3, 1, 5, 1, 1, 
  1
};
CONSTANT int filter_size_max = 5;

// input image of each convolution stage
CONSTANT int input_image_width[NUM_CONVOLUTIONS] =
{
  114,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};
CONSTANT int input_image_width_max = 114;

CONSTANT int input_image_height[NUM_CONVOLUTIONS] =
{
  114,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};
CONSTANT int input_image_height_max = 114;

// output image of each convolution stage
CONSTANT int output_image_width[NUM_CONVOLUTIONS] =
{
  112,
   56,
   56,
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
 };

CONSTANT int output_image_width_max = 112;

CONSTANT int output_image_height[NUM_CONVOLUTIONS] =
{
  112,
   56,
   56, 
   28, 28, 28, 28, 28, 28, 28, 
   28, 28, 28, 28, 28, 28, 28, 
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
   14, 14, 14, 14, 14, 14, 14,
    7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,
    1
};

CONSTANT int output_image_height_max = 112;

CONSTANT int input_channels[NUM_CONVOLUTIONS] =
{
  27,
  64,
  64,   
  192,  192,  96,  192,  16,  1,  192,   
  256,  256, 128,  256,  32,  1,  256,   
  480,  480,  96,  480,  16,  1,  480,   
  512,  512, 112,  512,  24,  1,  512,   
  512,  512, 128,  512,  24,  1,  512,   
  512,  512, 144,  512,  32,  1,  512,   
  528,  528, 160,  528,  32,  1,  528,   
  832,  832, 160,  832,  32,  1,  832,   
  832,  832, 192,  832,  48,  1,  832,
  1024
};

CONSTANT int output_channels[NUM_CONVOLUTIONS] =
{
   64,  
   64,
  192,   
   64,   96,  128,  16,  32,  192,  32,  
  128,  128,  192,  32,  96,  256,  64,  
  192,   96,  208,  16,  48,  480,  64,  
  160,  112,  224,  24,  64,  512,  64,  
  128,  128,  256,  24,  64,  512,  64,  
  112,  144,  288,  32,  64,  512,  64,  
  256,  160,  320,  32, 128,  528,  128,  
  256,  160,  320,  32, 128,  832,  128,  
  384,  192,  384,  48, 128,  832,  128,
  1000
};

CONSTANT int num_cvec[NUM_CONVOLUTIONS] =
{
  CEIL(27,   C_VECTOR),
  CEIL(64,   C_VECTOR),
  CEIL(64,   C_VECTOR),  
  CEIL(192,  C_VECTOR),  CEIL(192, C_VECTOR), CEIL(96,  C_VECTOR),  CEIL(192, C_VECTOR),  CEIL(16,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(192, C_VECTOR),  
  CEIL(256,  C_VECTOR),  CEIL(256, C_VECTOR), CEIL(128, C_VECTOR),  CEIL(256, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(256, C_VECTOR),  
  CEIL(480,  C_VECTOR),  CEIL(480, C_VECTOR), CEIL(96,  C_VECTOR),  CEIL(480, C_VECTOR),  CEIL(16,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(480, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(112, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(24,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(128, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(24,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(512,  C_VECTOR),  CEIL(512, C_VECTOR), CEIL(144, C_VECTOR),  CEIL(512, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(512, C_VECTOR),  
  CEIL(528,  C_VECTOR),  CEIL(528, C_VECTOR), CEIL(160, C_VECTOR),  CEIL(528, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(528, C_VECTOR),  
  CEIL(832,  C_VECTOR),  CEIL(832, C_VECTOR), CEIL(160, C_VECTOR),  CEIL(832, C_VECTOR),  CEIL(32,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(832, C_VECTOR),  
  CEIL(832,  C_VECTOR),  CEIL(832, C_VECTOR), CEIL(192, C_VECTOR),  CEIL(832, C_VECTOR),  CEIL(48,  C_VECTOR), CEIL(1, C_VECTOR),  CEIL(832, C_VECTOR),  
  CEIL(1024, C_VECTOR)  // fc1000
};
CONSTANT int num_cvec_max = CEIL(1024, C_VECTOR);

CONSTANT int num_cvec_filter[NUM_CONVOLUTIONS] =
{
 CEIL(27,  C_VECTOR),
 CEIL(64,  C_VECTOR*S_VECTOR),
 CEIL(64,  C_VECTOR),  
 CEIL(192, C_VECTOR*S_VECTOR), CEIL(192, C_VECTOR*S_VECTOR), CEIL(96,  C_VECTOR), CEIL(192, C_VECTOR*S_VECTOR),  CEIL(16, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(192, C_VECTOR*S_VECTOR),  
 CEIL(256, C_VECTOR*S_VECTOR), CEIL(256, C_VECTOR*S_VECTOR), CEIL(128, C_VECTOR), CEIL(256, C_VECTOR*S_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(256, C_VECTOR*S_VECTOR),  
 CEIL(480, C_VECTOR*S_VECTOR), CEIL(480, C_VECTOR*S_VECTOR), CEIL(96,  C_VECTOR), CEIL(480, C_VECTOR*S_VECTOR),  CEIL(16, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(480, C_VECTOR*S_VECTOR),  
 CEIL(512, C_VECTOR*S_VECTOR), CEIL(512, C_VECTOR*S_VECTOR), CEIL(112, C_VECTOR), CEIL(512, C_VECTOR*S_VECTOR),  CEIL(24, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(512, C_VECTOR*S_VECTOR),  
 CEIL(512, C_VECTOR*S_VECTOR), CEIL(512, C_VECTOR*S_VECTOR), CEIL(128, C_VECTOR), CEIL(512, C_VECTOR*S_VECTOR),  CEIL(24, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(512, C_VECTOR*S_VECTOR),  
 CEIL(512, C_VECTOR*S_VECTOR), CEIL(512, C_VECTOR*S_VECTOR), CEIL(144, C_VECTOR), CEIL(512, C_VECTOR*S_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(512, C_VECTOR*S_VECTOR),  
 CEIL(528, C_VECTOR*S_VECTOR), CEIL(528, C_VECTOR*S_VECTOR), CEIL(160, C_VECTOR), CEIL(528, C_VECTOR*S_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(528, C_VECTOR*S_VECTOR),  
 CEIL(832, C_VECTOR*S_VECTOR), CEIL(832, C_VECTOR*S_VECTOR), CEIL(160, C_VECTOR), CEIL(832, C_VECTOR*S_VECTOR),  CEIL(32, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(832, C_VECTOR*S_VECTOR),  
 CEIL(832, C_VECTOR*S_VECTOR), CEIL(832, C_VECTOR*S_VECTOR), CEIL(192, C_VECTOR), CEIL(832, C_VECTOR*S_VECTOR),  CEIL(48, C_VECTOR), CEIL(1, C_VECTOR*S_VECTOR),  CEIL(832, C_VECTOR*S_VECTOR),  
 CEIL(1024, C_VECTOR * S_VECTOR)  // fc1000
};
CONSTANT int num_cvec_filter_max = CEIL(1024, C_VECTOR*S_VECTOR);

// input
CONSTANT int END_WW_MAX_INPUT_READER = CEIL(114, S_VECTOR);

CONSTANT int num_kvec[NUM_CONVOLUTIONS] =
{
  CEIL(64,   K_VECTOR),
  CEIL(64,  K_VECTOR),
  CEIL(192,   K_VECTOR), 
  CEIL( 64,  K_VECTOR),  CEIL( 96,  K_VECTOR), CEIL(128,  K_VECTOR), CEIL(16,  K_VECTOR), CEIL( 32,  K_VECTOR), CEIL(192,   K_VECTOR), CEIL(32,   K_VECTOR), 
  CEIL(128,  K_VECTOR),  CEIL(128,  K_VECTOR), CEIL(192,  K_VECTOR), CEIL(32,  K_VECTOR), CEIL( 96,  K_VECTOR), CEIL(256,   K_VECTOR), CEIL(64,   K_VECTOR), 
  CEIL(192,  K_VECTOR),  CEIL( 96,  K_VECTOR), CEIL(208,  K_VECTOR), CEIL(16,  K_VECTOR), CEIL( 48,  K_VECTOR), CEIL(480,   K_VECTOR), CEIL(64,   K_VECTOR), 
  CEIL(160,  K_VECTOR),  CEIL(112,  K_VECTOR), CEIL(224,  K_VECTOR), CEIL(24,  K_VECTOR), CEIL( 64,  K_VECTOR), CEIL(512,   K_VECTOR), CEIL(64,   K_VECTOR), 
  CEIL(128,  K_VECTOR),  CEIL(128,  K_VECTOR), CEIL(256,  K_VECTOR), CEIL(24,  K_VECTOR), CEIL( 64,  K_VECTOR), CEIL(512,   K_VECTOR), CEIL(64,   K_VECTOR), 
  CEIL(112,  K_VECTOR),  CEIL(144,  K_VECTOR), CEIL(288,  K_VECTOR), CEIL(32,  K_VECTOR), CEIL( 64,  K_VECTOR), CEIL(512,   K_VECTOR), CEIL(64,   K_VECTOR), 
  CEIL(256,  K_VECTOR),  CEIL(160,  K_VECTOR), CEIL(320,  K_VECTOR), CEIL(32,  K_VECTOR), CEIL(128,  K_VECTOR), CEIL(528,   K_VECTOR), CEIL(128,  K_VECTOR), 
  CEIL(256,  K_VECTOR),  CEIL(160,  K_VECTOR), CEIL(320,  K_VECTOR), CEIL(32,  K_VECTOR), CEIL(128,  K_VECTOR), CEIL(832,   K_VECTOR), CEIL(128,  K_VECTOR), 
  CEIL(384,  K_VECTOR),  CEIL(192,  K_VECTOR), CEIL(384,  K_VECTOR), CEIL(48,  K_VECTOR), CEIL(128,  K_VECTOR), CEIL(832,   K_VECTOR), CEIL(128,  K_VECTOR), 
  CEIL(1000, K_VECTOR)  // fc1000
};
CONSTANT int num_kvec_max = CEIL(1000, K_VECTOR);

// used in pool
CONSTANT int begin_k[NUM_CONVOLUTIONS] =
{
    0,
    0, 
    0, 
    0,   0,    64,   0,  192,    0,   224, 
    0,   0,   128,   0,  320,    0,   416, 
    0,   0,   192,   0,  400,    0,   448, 
    0,   0,   160,   0,  384,    0,   448, 
    0,   0,   128,   0,  384,    0,   448, 
    0,   0,   112,   0,  400,    0,   464, 
    0,   0,   256,   0,  576,    0,   704, 
    0,   0,   256,   0,  576,    0,   704, 
    0,   0,   384,   0,  768,    0,   896, 
    0  // fc1000
};

// used in pool
CONSTANT int end_k[NUM_CONVOLUTIONS] =
{
   64, 
   64,
  192,   
   64,   96,  192,  16,  224,  192,   256,  
  128,  128,  320,  32,  416,  256,   480,  
  192,   96,  400,  16,  448,  480,   512,  
  160,  112,  384,  24,  448,  512,   512,  
  128,  128,  384,  24,  448,  512,   512,  
  112,  144,  400,  32,  464,  512,   528,  
  256,  160,  576,  32,  704,  528,   832,  
  256,  160,  576,  32,  704,  832,   832,  
  384,  192,  768,  48,  896,  832,  1024,
  1000   // fc1000
};

CONSTANT bool bias_enable[NUM_CONVOLUTIONS] = 
{
  1,
  1,
  1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 1, 0, 1,
  1
};

CONSTANT bool bn_enable[NUM_CONVOLUTIONS] = 
{
  1,
  0,
  1,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0,
  0
};

CONSTANT int ipool_enable[NUM_CONVOLUTIONS] = 
{
  0,
  0,
  0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 1, 0,
  0
};

CONSTANT int input_layer[NUM_CONVOLUTIONS] =
{
   0, 
   1,
   2,
   3,   3,   5,   3,   7,   3,   9,
  68,  68,  12,  68,  14,  68,  16,
  69,  69,  19,  69,  21,  69,  23,
  70,  70,  26,  70,  28,  70,  30,
  71,  71,  33,  71,  35,  71,  37,
  72,  72,  40,  72,  42,  72,  44,
  73,  73,  47,  73,  49,  73,  51,
  74,  74,  54,  74,  56,  74,  58,
  75,  75,  61,  75,  63,  75,  65,
  76
};

CONSTANT bool res_tail[NUM_CONVOLUTIONS] ={
  0,
  0, 
  0, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1, 
  1, 0, 1, 0, 1, 0, 1,
  0
};

CONSTANT int inception_layer[NUM_CONVOLUTIONS] ={
  0,
  0, 
  0, 
  0, 0, 0, 0, 0, 0, 0, 
  1, 0, 1, 0, 1, 0, 1, 
  2, 0, 2, 0, 2, 0, 2, 
  3, 0, 3, 0, 3, 0, 3, 
  4, 0, 4, 0, 4, 0, 4, 
  5, 0, 5, 0, 5, 0, 5, 
  6, 0, 6, 0, 6, 0, 6, 
  7, 0, 7, 0, 7, 0, 7, 
  8, 0, 8, 0, 8, 0, 8, 
  0 
};

// The below code will remove later.
CONSTANT int wait_after_conv[NUM_CONVOLUTIONS] = {
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

#endif // __CNN_GOOGLENET__
