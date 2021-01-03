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

#define BIT_IS_SET(value, bit_num) (((value) & (1ULL << (bit_num))) != 0)

unsigned char LpmMult(unsigned char dataa, unsigned char datab) {
  unsigned char result = dataa * datab;

  return result;
}

char MUL(char feature_val, char filter_val) {
  char filter_ab = 0x0f & filter_val;
  
  unsigned char data_raw = LpmMult(feature_val, filter_ab);

  char dot_accum = (BIT_IS_SET(filter_val, 7)) ? -data_raw : data_raw;

  return dot_accum;
}
