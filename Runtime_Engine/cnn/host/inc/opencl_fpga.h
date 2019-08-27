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

#ifndef __OPENCL_FPGA_H__
#define __OPENCL_FPGA_H__

#include "includes.h"

class OpenCLFPGA {
public:
  OpenCLFPGA();
  bool init();
  void cleanup();

  void create_kernel(std::string name, bool has_infinite_loop);
  kernel_info_t find_kernel(std::string name);

  cl_command_queue input_queue;
  cl_command_queue filter_queue;
  cl_command_queue bias_bn_queue;
  cl_command_queue wait_after_conv_queue;
  cl_command_queue output_queue;

  cl_program program;
  cl_int status;
  cl_platform_id platform;
  cl_context context;
  cl_device_id device;
  std::vector<kernel_info_t> kernels;
};

#endif
