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

#include "../../host/inc/defines.h"

int MUL(char feature, char filter) {
  if (BIT_IS_SET(filter, 6)) {
    return 0;
  }
 
  if (BIT_IS_SET(filter, 7)) {
    feature = -feature;
  }

  filter = 0x1f & filter;
  int data = feature << filter;
 
  return data;
}

int DotProduct(char16 feature_values, char16 filter_values) {
  int dot_accum = 0; // change from long int to int

  dot_accum += MUL(feature_values.s0, filter_values.s0);
  dot_accum += MUL(feature_values.s1, filter_values.s1);
  dot_accum += MUL(feature_values.s2, filter_values.s2);
  dot_accum += MUL(feature_values.s3, filter_values.s3);
  dot_accum += MUL(feature_values.s4, filter_values.s4);
  dot_accum += MUL(feature_values.s5, filter_values.s5);
  dot_accum += MUL(feature_values.s6, filter_values.s6);
  dot_accum += MUL(feature_values.s7, filter_values.s7);
  dot_accum += MUL(feature_values.s8, filter_values.s8);
  dot_accum += MUL(feature_values.s9, filter_values.s9);
  dot_accum += MUL(feature_values.sa, filter_values.sa);
  dot_accum += MUL(feature_values.sb, filter_values.sb);
  dot_accum += MUL(feature_values.sc, filter_values.sc);
  dot_accum += MUL(feature_values.sd, filter_values.sd);
  dot_accum += MUL(feature_values.se, filter_values.se);
  dot_accum += MUL(feature_values.sf, filter_values.sf);

  return dot_accum;
}
