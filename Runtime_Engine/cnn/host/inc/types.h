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

#ifndef __TYPES_H__
#define __TYPES_H__
// -------------------------------------------------------------------------- //
// cnn_types.h:
//
// Define all the types, structs used in the code.
// -------------------------------------------------------------------------- //

typedef char real;
typedef int Mreal;
typedef short Sreal;

#define REALMAX 127
#define REALMIN -128
#define ALPHA_INFLAT 20
#define INFLAT 15
#define LOWER  (INFLAT - 1)
#define TRANS_INFLAT (1.0/(1<<INFLAT))

typedef struct {
  Mreal bias;
  Mreal alpha;
  Mreal beta;
}bias_bn_param_t;
CONSTANT bias_bn_param_t zero_bias_bn = {0};

#endif // __TYPES_H__
