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

// -------------------------------------------------------------------------- //
// archs.h:
//
// This file contains all configurable architecture related options.
// -------------------------------------------------------------------------- //

#define IMAGE_BATCH_SIZE 1

// number of processing unit, parallel processing of different output channels
#define K_VECTOR 16

// parallel processing of different input channels
#define C_VECTOR 16

#define Q_VECTOR 5

#define S_VECTOR 3

#define RELU_K_VECTOR 16

#define W_VECTOR ( S_VECTOR + Q_VECTOR - 1 )

// default: enabled
#ifndef DISABLE_INFINITE_LOOPS
#define ENABLE_INFINITE_LOOPS
#endif

#endif // __ARCHS_H__
