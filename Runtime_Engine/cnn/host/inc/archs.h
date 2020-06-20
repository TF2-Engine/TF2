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

#ifndef __ARCHS_H__
#define __ARCHS_H__

//------------------------------------------------------------------------------------//
// archs.h                                                                            //
// Scope: Used by device code                                                         //
// Function: Configures all architecture related options.                             //
//------------------------------------------------------------------------------------//

#define IMAGE_BATCH_SIZE 1

// the vector size of output channels
#define N_VECTOR 4

// the vector size of input channels
#define C_VECTOR 4

// the vector size of output feature map width
#define OW_VECTOR 5

// the vector size of filter width
#define FW_VECTOR 3

// the vector size of output channels for relu, pool, etc.
#define NARROW_N_VECTOR 4

// the vector size of input feature map / image width
#define W_VECTOR (FW_VECTOR + OW_VECTOR - 1)

// controls whether kernels using the AUTORUN macro are actually autorun
// default: enabled
#ifndef DISABLE_AUTORUN_KERNELS
#define USE_AUTORUN_KERNELS
#endif

// Controls whether kernels have an outer loop which is an infinite loop.
// default: enabled
#ifndef DISABLE_INFINITE_LOOPS
#define ENABLE_INFINITE_LOOPS
#endif

#endif // __ARCHS_H__
