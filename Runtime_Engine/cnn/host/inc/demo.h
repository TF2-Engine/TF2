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
#ifndef __DEMO_H__
#define __DEMO_H__

#include "includes.h"

typedef struct{
  int             index;
  std::string     label_name; 
} imagenet_content;

typedef struct{
  int             label_index;
  std::string     jpg_image_name;
} imagenet_label;

typedef struct{
  int label;
  float feature;
} stat_item;

// int top_labels[5] = { -1, -1, -1, -1, -1 };

class Demo {
public:
    Demo();
    bool Init();
    void Evaluation(int test_index);
    void Result(cl_ulong total_sequencer, int num_images, double total_time);
    // void Softmax(int n, char *q, real *output);
    template<typename T>
    void Softmax(int n, float (*scale)[2], T *output);
    void CleanUp();

    std::vector<imagenet_content> imagecontents;

    std::vector<imagenet_label> imagenet_labels;

    int top_labels[5];
    int top1;
    int top5;
    int top1score;
    int top5score;

private:
    bool LoadLabel_Demo();
    
    int label;
    float softmax_result[5];

};

#endif
