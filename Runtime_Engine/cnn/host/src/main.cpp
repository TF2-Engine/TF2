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

#include "includes.h"

int main(int argc, char **argv) {
  if(argc != 5) {
    INFO("USAGE:\n%s <model_file> <quantization_file> <image_file> <num_images>\n", argv[0]);
    return 1;
  }

  char *model_file = argv[1];
  char *q_file = argv[2];
  char *image_file = argv[3];
  int num_images = (int)*argv[4];

  OpenCLFPGA platform;
  if(!platform.init()) {
    return -1;
  }  

  NetWork network;
  if(!network.init(platform, model_file, q_file, image_file, num_images)) {
    return -1;
  }

  Runner runner(platform, network);
  runner.init();
  runner.run();

    
  // verification

  //verify(0, verify_file_name, network.q, network.output);
  evaluation(0, network.q, network.output, network.top_labels);

  // cleanup
  network.cleanup();
  platform.cleanup();

  return 0;
}
