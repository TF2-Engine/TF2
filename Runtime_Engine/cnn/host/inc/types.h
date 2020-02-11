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

#ifndef __CNN_TYPES_H__
#define __CNN_TYPES_H__

//------------------------------------------------------------------------------------//
// types.h                                                                            //
// Scope: Used by device code                                                         //
// Function: Defines all the types, structs, and channels used throughout the code.   //
//------------------------------------------------------------------------------------//

typedef char CHAR;
typedef int INT;

typedef char real;
typedef int Mreal;
typedef short Sreal;
#define REALMAX 127
#define REALMIN -128
#define ALPHA_INFLAT 20
#define INFLAT 15
#define LOWER  (INFLAT - 1)
#define TRANS_INFLAT (1.0/(1<<INFLAT))

// -------------------------------------------------------------------------- //
typedef struct {
  Mreal bias;
  Mreal alpha;
  Mreal beta;
} BiasBnParam;
CONSTANT BiasBnParam BiasBnZero = {0};

// the rest of the types and functions defined here are intended to be used only in OpenCL code
#ifdef OPENCL

typedef struct {
  real v[C_VECTOR];
} DotVector;

typedef struct { 
  real v[W_VECTOR];
} OutputWidthVector;
CONSTANT OutputWidthVector OutputWidthVectorZero = {{{0}}};

typedef struct {
  uchar layer;
  ushort c_vec;
  
  bool padding[W_VECTOR + FW_VECTOR];
  short h;
  short feature_read_w_vec_header;
  short feature_read_w_inc_header;

  bool conv_start;
  bool conv_done[W_VECTOR];

  bool filter_loading;
  int filter_read_addr;
  char filter_read_fw_vec;
  bool filter_read_page;
  bool filter_loading_conv_idle;
  
  //bool kCacheWriteEnable;
  //bool write_ddr_enalbe;
} SequencerOutput;

typedef struct {
  DotVector v[FW_VECTOR];
} DotFilterVector;

typedef struct {
  //char pe_feature_scale[W_VECTOR][C_VECTOR_GROUP];
  DotVector v[W_VECTOR];
} DotFeatureVector;

typedef struct {
  DotVector data[W_VECTOR];
} InputReaderOutput;
CONSTANT InputReaderOutput InputReaderOutputZero = {{{{{0}}}}};

typedef struct {
  OutputWidthVector data;
  bool pe_output_relu;
  bool is_QVECTOR;
} PeOutput;

typedef struct {
  real v[NARROW_N_VECTOR];
} ReluChannelVector;
CONSTANT ReluChannelVector ReluChannelVectorZero = {{{0}}};

typedef struct {
  DotFilterVector filter_data;
  BiasBnParam bias_bn_data;
  int cache_addr;
  int n_inc;
} FilterReaderOutput;
CONSTANT FilterReaderOutput FilterReaderOutputZero = {{{{{{0}}}}}};

typedef struct {
  bool is_QVECTOR;
  bool conv_start;
  bool conv_done[W_VECTOR];
  bool pe_output_relu;

  // filter related signal
  int filter_read_addr;
  int filter_write_addr;
  char filter_read_fw_vec;
  bool filter_bias_read_page;
} PeControlSignal;
CONSTANT PeControlSignal PeControlSignalZero = {{0}};

typedef struct {
  DotFeatureVector input_data;
  bool input_data_valid;
} PeInputData;
CONSTANT PeInputData PeInputDataZero = {{{{{{0}}}}}};

typedef struct {
  DotFilterVector filter_data;
  BiasBnParam bias_bn_data;
  int n_inc;
  bool data_valid;
} PeInputFilter;
CONSTANT PeInputFilter PeInputFilterZero = {{{{{{0}}}}}};

typedef struct {
  OutputWidthVector data[NARROW_N_VECTOR];
} ReluOutput;
CONSTANT ReluOutput ReluOutputZero = {{{{{0}}}}};

typedef struct {
  OutputWidthVector data[NARROW_N_VECTOR]; 
} PoolOutput;

typedef struct {
  int cache_write_addr;
  real write_data[NEXT_POWER_OF_2(W_VECTOR)][NEXT_POWER_OF_2(C_VECTOR)];
} PoolTailOutput;
CONSTANT PoolTailOutput PoolTailOutputZero = {{{{0}}}};

// input data read from DDR
channel InputReaderOutput input_reader_output_channel __attribute__((depth(1)));

// filter data read from DDR
channel FilterReaderOutput filter_reader_output_channel __attribute__((depth(16)));

// the starting indices of current feature map that is being computed
channel SequencerOutput sequencer_output_channel __attribute__((depth(16)));

// input data, filter data, and control information sent to convolution kernels
channel PeInputData pe_input_data_channel_first __attribute__((depth(16)));
channel PeInputData pe_input_data_channel[N_VECTOR] __attribute__((depth(0)));

channel PeInputFilter pe_input_filter_channel_first __attribute__((depth(16)));
//channel PeInputFilter pe_input_filter_channel[N_VECTOR]    __attribute__((depth(8)));
channel PeInputFilter pe_input_filter_channel[N_VECTOR] __attribute__((depth(0)));

channel PeControlSignal pe_control_channel_first __attribute__((depth(16)));
channel PeControlSignal pe_control_channel[N_VECTOR] __attribute__((depth(0)));

// PE output data sent from PE kernels
channel PeOutput pe_output_channel[N_VECTOR] __attribute__((depth( 16 )));
channel PeOutput pe_drain_output_channel[N_VECTOR] __attribute__((depth( 1 )));

// the output of relu
channel ReluOutput relu_output_channel __attribute__((depth(1)));

// ipool channel
channel ReluOutput ipool_channel __attribute__((depth(1)));

/*
channel PoolOutput pool_output_channel  __attribute__((depth(256)));
channel PoolOutput end_pool_input_channel  __attribute__((depth(256)));
channel PoolTailOutput end_pool_output_channel  __attribute__((depth(256)));
channel PoolTailOutput feature_writer_input_channel __attribute__((depth(256)));
channel PoolTailOutput retriever_input_channel __attribute__((depth(256)));
//channel PoolTailOutput retriever_input_channel __attribute__((depth(64)));
*/
channel PoolOutput pool_output_channel __attribute__((depth(16)));
channel PoolTailOutput end_pool_input_channel __attribute__((depth(16)));
channel PoolTailOutput end_pool_output_channel __attribute__((depth(16)));
channel PoolTailOutput feature_writer_input_channel __attribute__((depth(0)));
channel PoolTailOutput retriever_input_channel __attribute__((depth(256)));

#endif // OPENCL

#endif // __TYPES_H__
