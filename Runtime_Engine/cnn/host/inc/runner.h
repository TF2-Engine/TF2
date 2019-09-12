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

#ifndef __RUNNER_H__
#define __RUNNER_H__

#include "includes.h"

class Runner {
public:
  Runner(OpenCLFPGA &platform, NetWork &network);

  void Init();

  void Run();

private:
  void EnqueueKernels(bool create_events, bool enqueue_infinite_loop);
  void WaitForAllKernels();
  void ReleaseAllEvents();
  
  OpenCLFPGA platform;
  NetWork network;
  char *image_file;
  int num_images;
};

#endif
