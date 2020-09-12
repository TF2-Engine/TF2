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

#ifndef __NETWORK_HELPER_H__
#define __NETWORK_HELPER_H__

#include "includes.h"

typedef struct {
  int label;
  float feature;
} StatItem;

typedef enum {
  kResnet50 = 0,
  kGooglenet= 1,
  kYandi = 2,
  kNil = 3
} NetworkType;

void Verify(int n, char *file_name, char *q, real *output);
void Evaluation(int n, char *q, real* output, int* top_labels);
void LoadLabel(int Num,int *labels);

// Doing sigmoid operation and write output to fp.
void SigmoidOutput(int n, char *q, real* output, FILE * fp);

// Reyurn network type.
NetworkType getNetwork();

#endif
