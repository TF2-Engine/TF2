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


#ifndef __YANDI__
#define __YANDI__

//--------------------------------------------------------------------//
//                                                                    //
//             KERNEL/HOST COMMON CNN PARAMETERS                      //
//                                                                    //
//                                                                    //
//--------------------------------------------------------------------//


//
// Debug Parameters
//


//#define CONCAT_LAYER_DEBUG

#define STATIC_CYCLE
#define PRINT_SEQUENCER_INDEX
//#define PRINT_N 1
// #define PRINT_POOL_INPUT
//#define PRINT_CYCLE


//
// Debug Parameters
//


//============================total summary===========================//
// number of convolution layers
#define NUM_LAYER (25) //for debug

#define NUM_CONVOLUTIONS (25)

#define NUM_Q_LAYERS (25 + 1 + 1)


#define INPUT_IMAGE_C 3

#define INPUT_IMAGE_H 256

#define INPUT_IMAGE_W 256

#define FIRST_FILTER_SIZE 3

#define MAX_OUT_CHANNEL 32

// the maximum pool window size

#define POOL_WINDOW_MAX 3

// set size of input_size
#define DDR_PAGE_SIZE0 (CEIL(32, C_VECTOR) * 256 * CEIL(256, W_VECTOR))
#define DDR_PAGE_SIZE1 (CEIL(32, C_VECTOR) * 256 * CEIL(256, W_VECTOR))
#define DDR_SIZE (DDR_PAGE_SIZE0 + DDR_PAGE_SIZE1)

#define CACHE_PAGE_SIZE (CEIL(32, C_VECTOR) * 256 * CEIL(256, W_VECTOR))
#define CACHE_SIZE (CACHE_PAGE_SIZE * 2)

// the largest of conv1 and conv2 filters
#define FILTER_CACHE_PAGE_SIZE1 (NEXT_DIVISIBLE(32, C_VECTOR) * 3 * CEIL(3, FW_VECTOR))
#define FILTER_CACHE_PAGE_SIZE2 (NEXT_DIVISIBLE(32, C_VECTOR) * 3 * CEIL(3, FW_VECTOR))
#define FILTER_CACHE_PAGE_SIZE  (MYMAX2(FILTER_CACHE_PAGE_SIZE1, FILTER_CACHE_PAGE_SIZE2))

///MODDDS
#define FILTER_CACHE_PAGE_DEPTH (NEXT_POWER_OF_2(CEIL(FILTER_CACHE_PAGE_SIZE, C_VECTOR)))
#define FILTER_CACHE_DEPTH (FILTER_CACHE_PAGE_DEPTH * DOUBLE_BUFFER_DIM)

#define FILTER_DDR_READ_STEP1 (CEIL((C_VECTOR * FW_VECTOR), DDR_BANDWIDTH_IN_BYTES))
#define FILTER_DDR_READ_STEP2 (CEIL((C_VECTOR * 1), DDR_BANDWIDTH_IN_BYTES))

#define FEATURE_DDR_READ_STEP 1

//Set size of host filter and bias buffer of each layer.
#define MAX_FILTER_SIZE2 (NEXT_POWER_OF_2(CEIL(32, C_VECTOR) * 3 * CEIL(3, FW_VECTOR) * NEXT_DIVISIBLE(32, N_VECTOR) * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))
#define MAX_FILTER_SIZE_TEMP (MAX_FILTER_SIZE2)
#define MAX_FILTER_SIZE (CEIL(MAX_FILTER_SIZE_TEMP, NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))

#define MAX_BIAS_SIZE  NEXT_DIVISIBLE(32, N_VECTOR)

#define NN_VEC (CEIL(N_VECTOR, NARROW_N_VECTOR))
#define DDR_BLOCK_SIZE DDR_PAGE_SIZE0
#define D0 0
#define D1 0
#define D2 DDR_BLOCK_SIZE

#define OUTPUT_OFFSET (2 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR))

#define C1 0
#define C2 CACHE_PAGE_SIZE
#define C3 (CACHE_PAGE_SIZE * 2)

//==========================basic structure===========================//
//----------------------------base addr info-----------------------------//
CONSTANT int kCacheReadBase[25] =
{
 C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1
};
CONSTANT int kCacheWriteBase[25] =
{
 C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2, C1, C2
};
CONSTANT int kDDRReadBase[25] =
{
 D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0
};
CONSTANT int kDDRWriteBase[25] =
{
 D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0, D0 + (CACHE_PAGE_SIZE / 2), D0, D0, D0, D0, D0, D0, D0
};
CONSTANT int kCacheWriteEnable[25] =
{
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
CONSTANT int kDDRWriteEnable[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
};
CONSTANT int kEndPoolEnable[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
CONSTANT int kAdditionEnable[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
};
CONSTANT int kBranchTail[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
CONSTANT int kAdditionReluEnable[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
CONSTANT int kIpoolEnable[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
CONSTANT bool kBiasEnable[25] =
{
 true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true
};
//----------------------------enable info-----------------------------//
CONSTANT bool kConvEnable[25] =
{
 true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true
};
CONSTANT bool kReluEnable[25] =
{
 true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false
};
CONSTANT bool kPoolEnable[25] =
{
 false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
};
CONSTANT bool kBnEnable[25] =
{
 false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
};

//----------------------concatenation and conv info-------------------//
CONSTANT int kConcatLayer[25] = {
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

CONSTANT int kInputLayer[25] =
{
 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
};

//-----------------------------wait cycles-------------------------------//
CONSTANT int kSequencerIdleCycle[25] = {
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

//===========================detail info===============================//
//-------------------------pool classification-------------------------//
// 0 - max pooling
// 1 - average pooling
CONSTANT int kPoolType[25] =
{
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

//-------------------------------size info----------------------------//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~conv~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//channel
CONSTANT int kCvecEnd[25] =
{
 CEIL(3, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(32, C_VECTOR)

};
CONSTANT int kCvecEndMax = CEIL(32, C_VECTOR);

CONSTANT int kFilterCvecEnd[25] =
{
 CEIL(3, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(16, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(8, C_VECTOR), CEIL(32, C_VECTOR)
};
CONSTANT int kFilterCvecEndMax = CEIL(32, C_VECTOR);

// input
CONSTANT int END_WW_MAX_INPUT_READER = CEIL(256, FW_VECTOR);

CONSTANT int kNvecEnd[25] =
{
 CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(16, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(8, N_VECTOR), CEIL(32, N_VECTOR), CEIL(1, N_VECTOR)
};
CONSTANT int kNvecEndMax = CEIL(32, N_VECTOR);

CONSTANT int kNEndWithOffset[25] =
{
 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 1
};
CONSTANT int kNEndWithOffsetMax = 32;

//input height
CONSTANT int kInputHeight[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int kInputHeightMax = 256;

//input width
CONSTANT int kInputWidth[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int kInputWidthMax = 256;

//output height
CONSTANT int kOutputHeight[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int kOutputHeightMax = 256;

//output width
CONSTANT int kOutputWidth[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int kOutputWidthMax = 256;

//This is the K dimension
CONSTANT int kOutputChannels[25] = {
 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 1
};
CONSTANT int kOutputChannelsMax = 32;

//This is the C dimension
CONSTANT int kInputChannels[25] = {
 3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32
};

//window size
CONSTANT int kFilterSize[25] = {
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};
CONSTANT int kFilterSizeMax = 3;

CONSTANT int kFWvecEnd[25] = {
 CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR), CEIL(3, FW_VECTOR)
};
CONSTANT int kFWvecEndMax = CEIL(3, FW_VECTOR);

CONSTANT int kWvecEnd[25] = {
 CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR)
};
CONSTANT int kWvecEndMax = CEIL(256, W_VECTOR);

//Conv pad
CONSTANT int kPoolPad[25] = {
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

CONSTANT int kConvStride[25] = {
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

//Conv pad
CONSTANT int kPadHeight[25] = {
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

CONSTANT int  kPadWidth[25] = {
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

// for pool computation
CONSTANT int kNStart[25] = {
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

CONSTANT int kNEnd[25] = {
 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 1
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~pool~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//pool output feature map width
CONSTANT int kOhEndWithOffset[25] = {
 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P, 256 + POOL_OFFSET_P
};
CONSTANT int kOhEndWithOffsetMax = 256 + POOL_OFFSET_P;

//pool output feature map hight
CONSTANT int kOwEndWithOffset[25] = {
 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q, 256 + POOL_OFFSET_Q
};
CONSTANT int kOwEndWithOffsetMax = 256 + POOL_OFFSET_Q;

CONSTANT int  kPoolOutputHeight[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int  kPoolOutputHeightMax = 256;

CONSTANT int  kPoolOutputWidth[25] = {
 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256
};
CONSTANT int kPoolOutputWidthMax = 256;

CONSTANT int kPoolOutputWvecEnd[25] = {
 CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR), CEIL(256, W_VECTOR)
};

CONSTANT int kPoolWindow[25] = {
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

CONSTANT bool kPoolStride2[25] = {
 false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
};

//----------------------------others----------------------------------//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~offset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// how much filter data (number of float_vec_t reads) we need to prefetch
// at each stage of convolution
// formula : num_cvec * R * Lnd_ss
CONSTANT int kFilterLoadSize[25] = {
 CEIL(3, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(16, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(8, C_VECTOR) * 3 * CEIL(3, FW_VECTOR), CEIL(32, C_VECTOR) * 3 * CEIL(3, FW_VECTOR)
};

//===========================cycles computation=======================//
//
// static cycles
//
#ifdef STATIC_CYCLE

CONSTANT int feature_writer_cycles[25] = {
 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 32768, 8192
};

CONSTANT int conv_cycles[25] = {
 66048, 132096, 132096, 132096, 132096, 132096, 132096, 132096, 132096, 132096, 132096, 132096, 66048, 33024, 33024, 33024, 33024, 33024, 33024, 33024, 33024, 33024, 33024, 132096, 132096
};

CONSTANT int pool_cycles[25] = {
 22188, 22188, 22188, 22188, 22188, 22188, 22188, 22188, 22188, 22188, 22188, 22188, 11094, 11094, 11094, 11094, 11094, 11094, 11094, 11094, 11094, 11094, 11094, 44376, 11094
};

CONSTANT int filter_reader_conv_cycles[25] = {
48, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 48, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96, 96
};

#define FEATURE_WRITER_CYCLE(i) feature_writer_cycles[i]
#define FILTER_READER_CONV_CYCLE(i) filter_reader_conv_cycles[i]
#define CONV_CYCLE(i) conv_cycles[i]
#define POOL_CYCLE(i) pool_cycles[i]

#define CONV_TOTAL_CYCLE 2179584

#define INPUT_READER_CYCLE 8192

#define FILTER_PRELOAD_CYCLE 24

#define FILTER_READER_CONV_TOTAL_CYCLE 1584

#define CONV_TOTAL_WRITE_CACHE 440320

#define POOL_TOTAL_CYCLE 443760

#define FEATURE_WRITER_TOTAL_CYCLE 327680

#define END_POOL_TOTAL_CYCLE 24

#endif

#ifndef STATIC_CYCLE


#define FEATURE_WRITER_CYCLE(i) FindFeatureWriterCycles(i)
#define FILTER_READER_CONV_CYCLE(i) FindFilterReaderConvCycles(i)
#define CONV_CYCLE(i) FindConvCycles(i)
#define POOL_CYCLE(i) FindPoolCycles(i)

#define CONV_TOTAL_CYCLE FindConvTotalCycles()

#define INPUT_READER_CYCLE FindInputReaderCycles()

#define FILTER_PRELOAD_CYCLE FindFilterPreloadCycles()

#define FILTER_READER_CONV_TOTAL_CYCLE FindFilterReaderConvTotalCycles()

#define CONV_TOTAL_WRITE_CACHE FindConvTotalWriteCache()

#define POOL_TOTAL_CYCLE FindPoolTotalCycles()

#define FEATURE_WRITER_TOTAL_CYCLE FindFeatureWriterTotalCycles()

#define END_POOL_TOTAL_CYCLE FindEndPoolTotalCycles()

#endif
#endif // __YANDI__

