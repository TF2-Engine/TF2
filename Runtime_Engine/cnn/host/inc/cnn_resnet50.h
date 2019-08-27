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

#ifndef __CNN_RESNET50__
#define __CNN_RESNET50__

//-----------------------------------------------------------------------                                         
//                                                                     
//CNN Network Ralated Paramters                  
//                                                                     
//-----------------------------------------------------------------------

// number of convolution layers
#define NUM_LAYER 54 
#define NUM_CONVOLUTIONS 54
#define NUM_Q_LAYERS ( NUM_CONVOLUTIONS + 1 ) 

#define INPUT_IMAGE_C 3
#define INPUT_IMAGE_H 224
#define INPUT_IMAGE_W 224

#define MAX_OUT_CHANNEL 2048

// set size of input_size
#define DDR_PAGE_SIZE0 (CEIL(256, C_VECTOR) * 56 * CEIL(56, W_VECTOR)) 
#define DDR_PAGE_SIZE1 (CEIL(256, C_VECTOR) * 56 * CEIL(56, W_VECTOR)) 
#define DDR_SIZE (DDR_PAGE_SIZE0 + DDR_PAGE_SIZE1)

#define CACHE_PAGE_SIZE (CEIL(256, C_VECTOR) * 56 * CEIL(56, W_VECTOR)) // single buffer size: be calculated from the layer of max slice size
#define CACHE_SIZE (CACHE_PAGE_SIZE * 2)

#define MAX_FILTER_SIZE1 (NEXT_POWER_OF_2(CEIL(2048, C_VECTOR) * 1 * CEIL(1, S_VECTOR) * NEXT_DIVISIBLE(2048, K_VECTOR) * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE2 (NEXT_POWER_OF_2(CEIL(1024, C_VECTOR) * 3 * CEIL(3, S_VECTOR) * NEXT_DIVISIBLE(1024, K_VECTOR) * NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE_TEMP ((MAX_FILTER_SIZE1 > MAX_FILTER_SIZE2) ? MAX_FILTER_SIZE1 : MAX_FILTER_SIZE2)
#define MAX_FILTER_SIZE (CEIL(MAX_FILTER_SIZE_TEMP, NEXT_POWER_OF_2(S_VECTOR * C_VECTOR)))

#define MAX_BIAS_SIZE  NEXT_DIVISIBLE(2048, K_VECTOR)

#define DDR_BLOCK_SIZE DDR_PAGE_SIZE0
#define D0 0
#define D1 0
#define D2 DDR_BLOCK_SIZE 

#define OUTPUT_OFFSET (2 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR))

CONSTANT int ddr_write_base[NUM_CONVOLUTIONS] = {
  D0,         // conv1
  D1,         // res2a_branch1
  D0, D0, D2, // res2a 
  D0, D0, D1, // res2b 
  D0, D0, D2, // res2c 
  D1,         // res3a_branch1
  D0, D0, D2, // res3a 
  D0, D0, D1, // res3b 
  D0, D0, D2, // res3c 
  D0, D0, D1, // res3d
  D2,         // res4a_branch1
  D0, D0, D1, // res4a 
  D0, D0, D2, // res4b 
  D0, D0, D1, // res4c 
  D0, D0, D2, // res4d 
  D0, D0, D1, // res4e 
  D0, D0, D2, // res4f
  D1,         // res5a_branch1
  D0, D0, D2, // res5a 
  D0, D0, D1, // res5b
  D0, D0, D2, // res5c 
  D0          // FC1000
};

CONSTANT bool res_tail[NUM_CONVOLUTIONS] ={
  0,
  0, 
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0,
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0  // fc1000
};

CONSTANT int filter_size[NUM_CONVOLUTIONS] =
{
  3, 
  1, 
  1, 3, 1, 
  1, 3, 1,
  1, 3, 1, 
  1,
  1, 3, 1,
  1, 3, 1, 
  1, 3, 1,
  1, 3, 1, 
  1, 
  1, 3, 1,
  1, 3, 1, 
  1, 3, 1,
  1, 3, 1, 
  1, 3, 1,
  1, 3, 1, 
  1, 
  1, 3, 1,
  1, 3, 1, 
  1, 3, 1,
  1, // fc1000
};
CONSTANT int filter_size_max = 3;

// input image of each convolution stage
CONSTANT int input_image_width[NUM_CONVOLUTIONS] =
{
  114,
  56,
  56, 56, 56, 
  56, 56, 56, 
  56, 56, 56,
  56,  
  56, 28, 28, 
  28, 28, 28, 
  28, 28, 28, 
  28, 28, 28,
  28,
  28, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14,
  14, 7,  7,
  7,  7,  7,
  7,  7,  7,
  1   // fc1000
};
CONSTANT int input_image_width_max = 114;

CONSTANT int input_image_height[NUM_CONVOLUTIONS] =
{
  114,
  56,
  56, 56, 56, 
  56, 56, 56, 
  56, 56, 56,
  56,  
  56, 28, 28, 
  28, 28, 28, 
  28, 28, 28, 
  28, 28, 28,
  28,
  28, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14, 14, 14,
  14,
  14, 7,  7,
  7,  7,  7,
  7,  7,  7,
  1   // fc1000
};
CONSTANT int input_image_height_max = 114;

// output image of each convolution stage
CONSTANT int output_image_width[NUM_CONVOLUTIONS] =
{
  112,
  56,
  56, 56, 56, 
  56, 56, 56,
  56, 56, 56,
  56,
  56, 28, 28,
  28, 28, 28, 
  28, 28, 28,
  28, 28, 28,
  28,
  28, 14, 14,
  14, 14, 14, 
  14, 14, 14,
  14, 14, 14, 
  14, 14, 14,
  14, 14, 14,
  14,
  14, 7,  7,
  7,  7,  7,  
  7,  7,  7,
  1   // fc1000
};

CONSTANT int output_image_width_max = 112;

CONSTANT int output_image_height[NUM_CONVOLUTIONS] =
{
  112,
  56,
  56, 56, 56, 
  56, 56, 56,
  56, 56, 56,
  56,
  56, 28, 28,
  28, 28, 28, 
  28, 28, 28,
  28, 28, 28,
  28,
  28, 14, 14,
  14, 14, 14, 
  14, 14, 14,
  14, 14, 14, 
  14, 14, 14,
  14, 14, 14,
  14,
  14, 7,  7,
  7,  7,  7,  
  7,  7,  7,
  1   // fc1000
};

CONSTANT int output_image_height_max = 112;

CONSTANT int input_channels[NUM_CONVOLUTIONS] =
{
  27, // pool1
  64,
  64,   64,     64, // res2a
  256,  64,     64, // res2b
  256,  64,     64, // res2c
  256,
  256,  128,    128,// res3a
  512,  128,    128,// res3b
  512,  128,    128,// res3c
  512,  128,    128,// res3d
  512,
  512,  256,    256,// res4a
  1024, 256,    256,// res4b
  1024, 256,    256,// res4c
  1024, 256,    256,// res4d
  1024, 256,    256,// res4e
  1024, 256,    256,// res4f
  1024,
  1024, 512,    512,// res5a
  2048, 512,    512,// res5b
  2048, 512,    512,// res5c
  2048  // fc1000
};

CONSTANT int output_channels[NUM_CONVOLUTIONS] =
{
  64, // pool1
  256,
  64, 64, 256, // res2a
  64, 64, 256, // res2b
  64, 64, 256, // res2c
  512,
  128, 128, 512, // res3a
  128, 128, 512, // res3b
  128, 128, 512, // res3c
  128, 128, 512, // res3d
  1024,
  256, 256, 1024, // res4a
  256, 256, 1024, // res4b
  256, 256, 1024, // res4c
  256, 256, 1024, // res4d
  256, 256, 1024, // res4e
  256, 256, 1024, // res4f
  2048,
  512, 512, 2048, // res5a
  512, 512, 2048, // res5b
  512, 512, 2048, // res5c
  1000   // fc1000
};
CONSTANT int output_channels_max = 2048;

CONSTANT bool ipool_enable[NUM_CONVOLUTIONS] =
{
  false,             // conv1
  false,             // res2a_branch1
  false, false, false, // res2a 
  false, false, false, // res2b 
  false, false, false, // res2c 
  false,             // res3a_branch1
  false, false, false, // res3a 
  false, false, false, // res3b 
  false, false, false, // res3c 
  false, false, false, // res3d
  false,             // res4a_branch1
  false, false, false, // res4a 
  false, false, false, // res4b 
  false, false, false, // res4c 
  false, false, false, // res4d 
  false, false, false, // res4e 
  false, false, false, // res4f
  false,             // res5a_branch1
  false, false, false, // res5a 
  false, false, false, // res5b
  false, false, false, // res5c 
  false             // fc1000
};

CONSTANT int input_layer[NUM_CONVOLUTIONS]=
{
	0,
	1 ,
	1 ,	3 ,	4 ,
	5,	6 ,	7 ,
	8 ,	9 ,	10,
	11,
	11,	13,	14,
	15,	16,	17,
	18,	19,	20,
	21,	22,	23,
	24,
	24,	26,	27,
	28,	29,	30,
	31,	32,	33,
	34,	35,	36,
	37,	38,	39,
	40,	41,	42,
	43,
	43,	45,	46,
	47,	48,	49,
	50,	51,	52,
	53
};

CONSTANT bool bias_enable[NUM_CONVOLUTIONS] =
{
	true,
	false,
	false,false,false,
	false,false,false,
	false,false,false,
  false,
	false,false,false,
	false,false,false,
	false,false,false,
	false,false,false,
 	false,
	false,false,false,
	false,false,false,
	false,false,false,
	false,false,false,
	false,false,false,
	false,false,false,
	false,
	false,false,false,
	false,false,false,
	false,false,false,
	true
};

CONSTANT int num_cvec[NUM_CONVOLUTIONS] =
{
  CEIL(27, C_VECTOR),
  CEIL(64, C_VECTOR),
  CEIL(64, C_VECTOR), CEIL(64, C_VECTOR), CEIL(64, C_VECTOR), 
  CEIL(256, C_VECTOR), CEIL(64, C_VECTOR), CEIL(64, C_VECTOR), 
  CEIL(256, C_VECTOR), CEIL(64, C_VECTOR), CEIL(64, C_VECTOR), 
  CEIL(256, C_VECTOR),
  CEIL(256, C_VECTOR), CEIL(128, C_VECTOR), CEIL(128, C_VECTOR),
  CEIL(512, C_VECTOR), CEIL(128, C_VECTOR), CEIL(128, C_VECTOR), 
  CEIL(512, C_VECTOR), CEIL(128, C_VECTOR), CEIL(128, C_VECTOR),
  CEIL(512, C_VECTOR), CEIL(128, C_VECTOR), CEIL(128, C_VECTOR),
  CEIL(512, C_VECTOR),
  CEIL(512, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR),
  CEIL(1024, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR), 
  CEIL(1024, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR),
  CEIL(1024, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR), 
  CEIL(1024, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR),
  CEIL(1024, C_VECTOR), CEIL(256, C_VECTOR), CEIL(256, C_VECTOR),
  CEIL(1024, C_VECTOR),
  CEIL(1024, C_VECTOR), CEIL(512, C_VECTOR), CEIL(512, C_VECTOR),
  CEIL(2048, C_VECTOR), CEIL(512, C_VECTOR), CEIL(512, C_VECTOR), 
  CEIL(2048, C_VECTOR), CEIL(512, C_VECTOR), CEIL(512, C_VECTOR),
  CEIL(2048, C_VECTOR)  // fc1000
};
CONSTANT int num_cvec_max = CEIL(2048, C_VECTOR);

CONSTANT int num_kvec[NUM_CONVOLUTIONS] =
{
  CEIL(64, K_VECTOR),
  CEIL(256, K_VECTOR),
  CEIL(64, K_VECTOR), CEIL(64, K_VECTOR), CEIL(256, K_VECTOR),
  CEIL(64, K_VECTOR), CEIL(64, K_VECTOR), CEIL(256, K_VECTOR),
  CEIL(64, K_VECTOR), CEIL(64, K_VECTOR), CEIL(256, K_VECTOR),
  CEIL(512, K_VECTOR),
  CEIL(128, K_VECTOR), CEIL(128, K_VECTOR), CEIL(512, K_VECTOR),
  CEIL(128, K_VECTOR), CEIL(128, K_VECTOR), CEIL(512, K_VECTOR),
  CEIL(128, K_VECTOR), CEIL(128, K_VECTOR), CEIL(512, K_VECTOR),
  CEIL(128, K_VECTOR), CEIL(128, K_VECTOR), CEIL(512, K_VECTOR),
  CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(256, K_VECTOR), CEIL(256, K_VECTOR), CEIL(1024, K_VECTOR),
  CEIL(2048, K_VECTOR),
  CEIL(512, K_VECTOR), CEIL(512, K_VECTOR), CEIL(2048, K_VECTOR),
  CEIL(512, K_VECTOR), CEIL(512, K_VECTOR), CEIL(2048, K_VECTOR),
  CEIL(512, K_VECTOR), CEIL(512, K_VECTOR), CEIL(2048, K_VECTOR),
  CEIL(1000, K_VECTOR)  // fc1000
};
CONSTANT int num_kvec_max = CEIL(2048, K_VECTOR);

CONSTANT int begin_k[NUM_CONVOLUTIONS] =
{
  0,
  0, 
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0,
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0, 
  0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, 0,
  0  // fc1000
};

CONSTANT int end_k[NUM_CONVOLUTIONS] =
{
  64, // pool1
  256,
  64, 64, 256, // res2a
  64, 64, 256, // res2b
  64, 64, 256, // res2c
  512,
  128, 128, 512, // res3a
  128, 128, 512, // res3b
  128, 128, 512, // res3c
  128, 128, 512, // res3d
  1024,
  256, 256, 1024, // res4a
  256, 256, 1024, // res4b
  256, 256, 1024, // res4c
  256, 256, 1024, // res4d
  256, 256, 1024, // res4e
  256, 256, 1024, // res4f
  2048,
  512, 512, 2048, // res5a
  512, 512, 2048, // res5b
  512, 512, 2048, // res5c
  1000   // fc1000
};

CONSTANT bool bn_enable[NUM_CONVOLUTIONS] = 
{
  1,
	1,
	1, 1, 1,
	1, 1, 1,
	1, 1, 1,
	1,
	1, 1, 1,
	1, 1, 1,
	1, 1, 1,
	1, 1, 1,
	1,
	1, 1, 1,
	1, 1, 1,
  1, 1, 1,
	1, 1, 1,
	1, 1, 1,
	1, 1, 1,
	1,
	1, 1, 1,
  1, 1, 1,
	1, 1, 1,
  0

};

CONSTANT int inception_layer[NUM_CONVOLUTIONS] ={0};

CONSTANT int wait_after_conv[NUM_CONVOLUTIONS] = { 
  1000, 
  0, 
  0, 0, 0,
  0, 0, 0, 
  0, 0, /*5000*/1000,
  0,
  0, 0, 0, 
  0, 0, 0,
  0, 0, 0,
  0, 0, /*5000*/1000,
  0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, /*5000*/1000,
  0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0
};

#endif // __CNN_RESNET50__
