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


#ifndef OPENCL
#define OPENCL
#endif

#include "../../host/inc/cnn.h"

// Functions:
// Computes the final cycles needed by serveral kernels when the STATIC_CYCLE macro is undefined.
// Warning: This file will be deprecated in future version.

#ifndef STATIC_CYCLE
//
// computes how many cycles it takes to preload the first filter
int FindFilterPreloadCycles() {
  return kFilterLoadSize[0] * N_VECTOR * FILTER_DDR_READ_STEP1;
}

// computes how many cycles it takes to process a OW_VECTOR / W_VECTOR sequencer_output pixels
int FindConvCvecCycles(int layer) {
  int FH = kFilterSize[layer];
  int C_VEC = kCvecEnd[layer];
  int FW_VEC = kFWvecEnd[layer];
  
  return C_VEC * FH * FW_VEC;
}

int FindConvNvecCycles(int layer, int n_vec, int N_VEC) {
  int filter_load_layer = layer;
  int OW = kOutputWidth[layer];
  int OH = CEIL(kOutputHeight[layer], kConvStride[layer]);
  int FH = kFilterSize[layer];

  int WOW_vec;
  if (FH != 1) {
    WOW_vec = CEIL(OW, OW_VECTOR);
  } else { // FH==1
    WOW_vec = CEIL(OW, W_VECTOR);
  }
  int OHxWOW_vec = OH * WOW_vec;

  int conv_cyc = OHxWOW_vec * FindConvCvecCycles(layer);
  if (n_vec == N_VEC - 1) {
    filter_load_layer = layer + 1;
  }

  int FILTER_DDR_READ_STEP = FH != 1 ? FILTER_DDR_READ_STEP1 : FILTER_DDR_READ_STEP2;
  int write_cyc = kFilterLoadSize[filter_load_layer] * N_VECTOR * FILTER_DDR_READ_STEP;
  if ((layer == NUM_LAYER - 1) && (n_vec == N_VEC - 1)) {
    return conv_cyc;
  } else {
    return conv_cyc > write_cyc ? conv_cyc : write_cyc;
  }
}

int FindConvCycles(int layer) {
  int total_cycles = 0;
  int temp[8] = {0};
  for (int i = 0; i < kNvecEnd[layer]; i++) {
    int sum = temp[7] + FindConvNvecCycles( layer, i, kNvecEnd[layer]);
    for (int j = 7; j > 0; j--) {
      temp[j] = temp[ j - 1 ];
    }
    temp[0] = sum;
  }

  for (int i = 0; i < 8; i++) {
    total_cycles += temp[i];
  }

  return total_cycles;
}

// computes the total number of cycles the sequencer will run for
int FindConvTotalCycles() {
  int total_cycles = 0;
  #pragma unroll  
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    total_cycles += FindConvCycles(layer);
  }
  
  return total_cycles;
}

int FindPeCycles(int layer) {
  int total_cycles = 0;

  if (kIpoolEnable[layer]) {
    if (layer != NUM_LAYER - 1) {
      int filter_load_layer = layer + 1;
      total_cycles = kFilterLoadSize[filter_load_layer] * N_VECTOR * FILTER_DDR_READ_STEP2;
    }
  } else {
    total_cycles = FindConvCycles(layer);
  }
  
  return total_cycles;
}

// computes the total number of cycles the pe will run for
int FindPeTotalCycles() {
  int total_cycles = 0;
  #pragma unroll  
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    total_cycles += FindPeCycles(layer);
  }
  return total_cycles;
}

// compute the total number of cycles the sequencer will run for
int FindConvLayerCycles(int total_layer) {
  int total_cycles = 0;
  #pragma unroll  
  for (int layer = 0; layer < total_layer; layer++) {
    total_cycles += FindConvCycles(layer);
  }
  return total_cycles;
}

int FindConvWriteCache(int layer) {
  int OW = kOutputWidth[layer];
  int OH = CEIL(kOutputHeight[layer], kConvStride[layer]);
  int FH = kFilterSize[layer];
  int N_VEC = kNvecEnd[layer];

  int WOW_VECTOR = FH != 1 ? OW_VECTOR : W_VECTOR;
  int OW_VEC = CEIL( OW, WOW_VECTOR );
  int OHxOW_VEC = OH * OW_VEC;

  return  N_VEC * OHxOW_VEC;
}

int FindConvTotalWriteCache() {
  int cycles = 0;
  #pragma unroll 
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    cycles += kIpoolEnable[layer] ? 0 : FindConvWriteCache( layer );
  }
  
  return cycles;
}

int FindPoolCycles(int layer) {
  int FW = kFilterSize[layer];
  int N = kNEndWithOffset[layer];

  int OH = kOhEndWithOffset[layer];
  int OW = kOwEndWithOffset[layer];

  int N_VEC = CEIL(N, N_VECTOR);
  
  int WOW_VECTOR = FW != 1 ? OW_VECTOR : W_VECTOR;
  
  int W_VEC = CEIL(OW, WOW_VECTOR);
   
  int NN_VEC = CEIL(N_VECTOR, NARROW_N_VECTOR);

  return N_VEC * OH * W_VEC * NN_VEC;
}

int FindPoolTotalCycles() {
  int total_cycles = 0;
  
  #pragma unroll 
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    total_cycles += FindPoolCycles(layer);
  }
  
  return total_cycles;
}

int FindFeatureWriterCycles(int layer) {
  int total_cycles = 0;

  int N = kOutputChannels[layer];
  int H = kPoolOutputHeight[layer];
  int W = kPoolOutputWidth[layer];       
  int FW = kFilterSize[layer];
  
  total_cycles = CEIL(N, N_VECTOR) * CEIL(N_VECTOR, NARROW_N_VECTOR) * H * CEIL(W, W_VECTOR);
    
  return total_cycles;
}

int FindFeatureWriterTotalCycles() {
  int total_cycles = 0;
  
  #pragma unroll 
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    total_cycles += FindFeatureWriterCycles(layer);
  }
  
  return total_cycles;
}

int FindEndPoolTotalCycles() {
  int total_cycles = 0;
  #pragma unroll 
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    if (kEndPoolEnable[layer])
      total_cycles += FindFeatureWriterCycles(layer);
  }
  
  return total_cycles;
}

// compute how many cycles it takes to read a single convolution layer filter
int FindFilterReaderConvCycles(int layer) {
  int N_VEC = kNvecEnd[layer];
  int FH = kFilterSize[layer];
  int C = kInputChannels[layer];
  int C_VEC = FH == 1 ? CEIL(C, C_VECTOR * FW_VECTOR) : CEIL(C, C_VECTOR);
  int FW_VEC = kFWvecEnd[layer];
  
  return kIpoolEnable[layer] ? 0 : N_VEC * C_VEC * FH * FW_VEC * N_VECTOR;
}

int FindFilterReaderConvTotalCycles() {
  int total_cycles = 0;
  
  #pragma unroll 
  for (int layer = 0; layer < NUM_LAYER; layer++) {
    total_cycles += FindFilterReaderConvCycles(layer);
  }
  
  return total_cycles;
}

int FindInputReaderCycles() {
  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  
  return CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR);
}

#endif