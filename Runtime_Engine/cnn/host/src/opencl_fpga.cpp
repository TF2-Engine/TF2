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

OpenCLFPGA::OpenCLFPGA() {

}

bool OpenCLFPGA::Init() {
  cl_int status;

  if (!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel");
  if (platform == NULL) {
    ERROR("Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  DisplayDeviceInfo(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // generate list of kernels based on macros
#ifdef ENABLE_INFINITE_LOOPS
  bool enable_infinite_loops = true;
#else
  bool enable_infinite_loops = false;
#endif

  create_kernel("input_reader", /* has_infinite_loop */ false);
  create_kernel("sequencer", /* has_infinite_loop */ false);
  create_kernel("retriever", /* has_infinite_loop */ false);
  create_kernel("filter_reader", /* has_infinite_loop */ false);

  for (int i = 0; i < N_VECTOR; i++) {
    char kernel_name[1024];
    sprintf(kernel_name,"%s%d", "pe_kernel_", i);
    //create_kernel(kernel_name, /* has_infinite_loop */ enable_infinite_loops);
  }

  //create_kernel("pe_tail", /* has_infinite_loop */ true);
  create_kernel("relu", /* has_infinite_loop */ enable_infinite_loops);
  create_kernel("pool", /* has_infinite_loop */ enable_infinite_loops);
  create_kernel("pool_tail", /* has_infinite_loop */ false);
  create_kernel("full_size_pool", /* has_infinite_loop */ true);
  create_kernel("feature_writer", /* has_infinite_loop */ false);

  // Create the command queue.
  for (int i = 0; i < kernels.size(); i++) {
    kernels[i].queue = clCreateCommandQueue(context, device,CL_QUEUE_PROFILING_ENABLE, &status);
    std::string err_msg = "Failed clCreateCommandQueue : " + kernels[i].name;
    checkError(status, err_msg.c_str());
  }

  input_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status,"Failed clCreateCommandQueue : input_queue");

  filter_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status,"Failed clCreateCommandQueue : filter_queue");

  bias_bn_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status,"Failed clCreateCommandQueue : bias_bn_queue");

  wait_after_conv_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status,"Failed clCreateCommandQueue : wait_after_conv_queue");

  output_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status,"Failed clCreateCommandQueue : output_queue");

  // Create the program.
  std::string binary_file = getBoardBinaryFile(COMPILED_BINARY, device);
  INFO("Using AOCX: %s\n", binary_file.c_str());
  std::string dir = "../device/";
  dir.append(binary_file);
  INFO("cnn.aocx dir=%s\n", dir.c_str());
  program = createProgramFromBinary(context, dir.c_str(), &device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  for (int i = 0; i < kernels.size(); i++) {
    kernels[i].kernel = clCreateKernel(program, kernels[i].name.c_str(), &status);
    std::string err_msg = "Failed clCreateKernel : " + kernels[i].name;
    checkError(status, err_msg.c_str());
  }

  return true;
}

void OpenCLFPGA::create_kernel(std::string name, bool has_infinite_loop) {
  KernelInfo kernel_info;
  kernel_info.name = name;
  kernel_info.has_infinite_loop = has_infinite_loop;
  kernel_info.queue = NULL;
  kernel_info.event = NULL;
  kernel_info.kernel = NULL;
  kernels.push_back(kernel_info);
}

KernelInfo OpenCLFPGA::find_kernel(std::string name) {
  for (int i = 0; i < kernels.size(); i++) {
    if (kernels[i].name == name) return kernels[i];
  }
  ERROR("Can't find kernel named %s at %s:%d\n", name.c_str());
  exit(1);
}

void OpenCLFPGA::CleanUp()
{
  for (int i = 0; i < kernels.size(); i++) {
    if (kernels[i].queue) clReleaseCommandQueue(kernels[i].queue);
    if (kernels[i].kernel) clReleaseKernel(kernels[i].kernel);
  }
  
  // free command queue
  if (input_queue) clReleaseCommandQueue(input_queue);
  if (filter_queue) clReleaseCommandQueue(filter_queue);
  if (bias_bn_queue) clReleaseCommandQueue(bias_bn_queue);
  if (wait_after_conv_queue) clReleaseCommandQueue(wait_after_conv_queue);
  if (output_queue) clReleaseCommandQueue(output_queue);

  // program and context
  if (context) clReleaseContext(context);
  if (program) clReleaseProgram(program);
}
