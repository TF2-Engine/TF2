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

NetWork::NetWork() {

}

bool NetWork::Init(OpenCLFPGA &platform, char *model_file, char* q_file, char *image_file, int num_images) {
  this->platform = platform;
  this->model_file = model_file;
  this->q_file = q_file;
  this->image_file = image_file;
  this->num_images = num_images;

  if (!(InitNetwork())) {
    // return -1;
    return false;
  }

  if (!(InitBuffer())) {
    // return -1;
    return false;
  }
  
  // return 0;
  return true;
}

bool NetWork::InitNetwork() {
  //
  // input array
  //

  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  int HXW = H * W;

  // const unsigned long long int input_raw_size = C * HXW * num_images;
  input_raw_size = C * HXW * num_images;

  input_raw = (float*)alignedMalloc(sizeof(float)* input_raw_size);

  // if (input_raw == NULL) printf("Cannot allocate enough space for input_raw\n");
  if (input_raw == NULL) {
    printf("Cannot allocate enough space for input_raw\n"); 
    return false;
  }
  memset(input_raw, 0, sizeof(float) * input_raw_size);

  INFO("Loading input image binary...\n");

  const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;
  input = (float*)alignedMalloc(sizeof(float) * input_device_size);
  // if (input == NULL) ERROR("Cannot allocate enough space for input\n");
  if (input == NULL) {
    ERROR("Cannot allocate enough space for input\n"); 
    return false;
  }
  memset(input, 0, sizeof(float) * input_device_size);

  input_real = (real*)alignedMalloc(sizeof(real) * input_device_size);
  // if (input_real == NULL) ERROR("Cannot allocate enough space for input_real\n");
  if (input_real == NULL) {
    ERROR("Cannot allocate enough space for input_real\n"); 
    return false;
  }

  sim_out = (float*)alignedMalloc(sizeof(float)* input_raw_size);

  if (sim_out == NULL) {
    ERROR("Cannot allocate enough space for input_raw\n");
    return false;
  }
  memset(sim_out, 0, sizeof(float) * input_raw_size);

  // filter array(incldue bias array)
  const int filter_raw_size =  NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
  // filter_raw = (real*)alignedMalloc(sizeof(real) * filter_raw_size);
  filter_raw = (int8*)alignedMalloc(sizeof(int8) * filter_raw_size);
  // if (filter_raw== NULL) ERROR("Cannot allocate enough space for filter_raw\n");
  if (filter_raw== NULL) {
    ERROR("Cannot allocate enough space for filter_raw\n"); 
    return false;
  }
  memset(filter_raw, 64, sizeof(int8) * filter_raw_size);

  const int bias_bn_size = NUM_CONVOLUTIONS * MAX_BIAS_SIZE;
  bias_bn = (BiasBnParam*)alignedMalloc(sizeof(BiasBnParam) * bias_bn_size);
  // if (bias_bn == NULL) ERROR("Cannot allocate enough space for bias_bn\n");
  if (bias_bn == NULL) {
    ERROR("Cannot allocate enough space for bias_bn\n");
    return false;
  }
  memset(bias_bn, 0, sizeof(BiasBnParam) * bias_bn_size);

  // compute Quantization param
  // it can compute only once
  // float *input_raw_images = (float*)malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);

  // q = (char *)alignedMalloc(sizeof(char) * NUM_Q_LAYERS * MAX_OUT_CHANNEL);
  // Quantization(q, input_raw_images, q_file);

  const int scale_size = NUM_CONVOLUTIONS + 1;
  // scale = (float(*)[3])alignedMalloc(sizeof(float)* scale_size * 3);
  scale = (float(*)[2])malloc(sizeof(float)* scale_size * 2);
  if (scale == NULL) {
    ERROR("Cannot allocate enough space for scale\n");
    return false;
  }
  memset(scale, 0, sizeof(float) * scale_size * 2);


  const int feature_ddr_size = OUTPUT_OFFSET + num_images * OUTPUT_OFFSET;
  output_pre = (float*)malloc(sizeof(float) * feature_ddr_size );
  if (output_pre == NULL) {
    ERROR("Cannot allocate enough space for output_pre\n");
    return false;
  }
  memset(output_pre, 0.0, sizeof(float) * feature_ddr_size);
  // Quantization(scale, input_raw_images, q_file);
  Quantization(scale, q_file);
  
  INFO("Loading convolutional layer params...\n");
  //LoadModel(model_file, filter_raw, bias,alpha,beta, q);
  // LoadModel(model_file, filter_raw, bias_bn, q);
  // LoadModel(model_file, q_file, filter_raw, bias_bn, scale);
    LoadModel(model_file, filter_raw, bias_bn, scale);
  
  const int filter_device_size = NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
  filter = (real*)alignedMalloc(sizeof(real) * filter_device_size);
  if (filter == NULL) {
    ERROR("Cannot allocate enough space for filter.\n");
    return false;
  }

  filter_real = (real*)alignedMalloc(sizeof(real) * filter_device_size);
  if (filter_real == NULL) {
    ERROR("Cannot allocate enough space for filter_real.\n");
    return false;
  }
  memset(filter_real, 64, sizeof(real) * filter_device_size);

  FilterConvert(filter, filter_raw, filter_real);

  return true;
}

bool NetWork::InitBuffer() { 
  // output array
  //const int feature_ddr_size = DDR_SIZE * NEXT_POWER_OF_2(W_VECTOR) * NEXT_POWER_OF_2(C_VECTOR);
  const int feature_ddr_size = OUTPUT_OFFSET + num_images * OUTPUT_OFFSET;
  output = (real*)alignedMalloc(sizeof(real) * feature_ddr_size );
  if (output == NULL) {
    ERROR("Cannot allocate enough space for output\n");
    return false;
  }

  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  int HXW = H * W;
  
  cl_int status;

  const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;
  const int filter_device_size = NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
  
  // feature_ddr
  feature_ddr = clCreateBuffer(platform.context, CL_MEM_READ_WRITE, sizeof(real) * feature_ddr_size, NULL, &status);
  checkError(status, "Failed clCreateBuffer : feature_ddr");
  
  // input_buffer
  input_buffer[0] = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(real)* input_device_size, NULL, &status);
  checkError(status, "Failed clCreateBuffer : input_buffer[0]");
  input_buffer[1] = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(real)* input_device_size, NULL, &status);
  checkError(status, "Failed clCreateBuffer : input_buffer[1]");

  // filter_buffer
  filter_buffer = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(real) * filter_device_size, NULL, &status);
  checkError(status,"Failed clCreateBuffer : filter_buffer");
  status = clEnqueueWriteBuffer(platform.filter_queue, filter_buffer, CL_TRUE, 0, sizeof(real) * filter_device_size, filter_real, 0, NULL, NULL);
  checkError(status,"Failed clEnqueueWriteBuffer : filter_queue");
  clFinish(platform.filter_queue);

  // bias_bn buffer
  const int bias_bn_size = NUM_CONVOLUTIONS * MAX_BIAS_SIZE;
  bias_bn_buffer = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(BiasBnParam) * bias_bn_size, NULL, &status);
  checkError(status,"Failed clCreateBuffer : bias_bn_buffer");
  status = clEnqueueWriteBuffer(platform.bias_bn_queue, bias_bn_buffer, CL_TRUE, 0, sizeof(BiasBnParam) * bias_bn_size, bias_bn, 0, NULL, NULL);
  checkError(status,"Failed clEnqueueWriteBuffer : bias_bn_queue");
  clFinish(platform.bias_bn_queue);

  // const int scale_size = NUM_CONVOLUTIONS;
  // scale_buffer = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(float)* scale_size * 3, NULL, &status);
  // checkError(status,"Failed clCreateBuffer : scale_buffer");
  // status = clEnqueueWriteBuffer(platform.scale_queue, scale_buffer, CL_TRUE, 0, sizeof(float)* scale_size * 3, scale, 0, NULL, NULL);
  // checkError(status,"Failed clEnqueueWriteBuffer : scale_queue");
  // clFinish(platform.scale_queue);

  // wait_after_conv_cycles buffer
  wait_after_conv_cycles = clCreateBuffer(platform.context, CL_MEM_READ_ONLY, sizeof(int) * NUM_CONVOLUTIONS, NULL, &status);
  checkError(status,"Failed clCreateBuffer : wait_after_conv_cycles");
  status = clEnqueueWriteBuffer(platform.wait_after_conv_queue, wait_after_conv_cycles, CL_TRUE, 0, sizeof(int) * NUM_CONVOLUTIONS, &kSequencerIdleCycle[0], 0, NULL, NULL);
  checkError(status,"Failed clEnqueueWriteBuffer : kSequencerIdleCycle");
  clFinish(platform.wait_after_conv_queue);  
  
  // return 0; 
  return true; 
}

void NetWork::CleanUp() {
  // mem objects
  if (input_buffer[0]) clReleaseMemObject(input_buffer[0]);
  if (input_buffer[1]) clReleaseMemObject(input_buffer[1]);
  if (feature_ddr) clReleaseMemObject(feature_ddr);
  if (filter_buffer) clReleaseMemObject(filter_buffer);
  if (bias_bn_buffer) clReleaseMemObject(bias_bn_buffer);
  if (wait_after_conv_cycles) clReleaseMemObject(wait_after_conv_cycles);

  // host buffers
  if (input_raw) alignedFree(input_raw);
  if (input) alignedFree(input);
  if (q) alignedFree(q);
  if (filter_raw) alignedFree(filter_raw);
  if (filter) alignedFree(filter);
  if (bias_bn) alignedFree(bias_bn); 
  if (sim_out) alignedFree(sim_out);
  if (scale) free(scale);
  if (output_pre) free(output_pre);
}
