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

#include "runner.h"

Runner::Runner(OpenCLFPGA &platform, NetWork &network) {
  this->platform = platform;
  this->network = network;
}

void Runner::Init() {
  // this->image_file = network.image_file;
  this->num_images = network.num_images; 
}

void Runner::EnqueueKernels(bool create_events, bool enqueue_infinite_loop) {
  for (int i = 0; i < platform.kernels.size(); i++) {
    if (platform.kernels[i].has_infinite_loop == enqueue_infinite_loop) {
      //INFO("infinite_loop=%d kernel_name=%s\n", enqueue_infinite_loop, platform.kernels[i].name.c_str());
      cl_int status = clEnqueueTask(platform.kernels[i].queue, platform.kernels[i].kernel, 0, NULL,
          create_events ? &(platform.kernels[i]).event : NULL);
      std::string err_msg = platform.kernels[i].name + ": Failed to launch kernel.";
      checkError(status, err_msg.c_str());
    }
  }
}

void Runner::WaitForAllKernels() {
  for (int i = 0; i < platform.kernels.size(); i++) {
    if (!platform.kernels[i].has_infinite_loop) {
      clFinish(platform.kernels[i].queue);
    }
  }
}

void Runner::ReleaseAllEvents() {
  for (int i = 0; i < platform.kernels.size(); i++) {
    clReleaseEvent(platform.kernels[i].event);
  }
}

// void Runner::Run() {
void Runner::Run(char *image_file) {
  cl_int status;
  
  //
  // set args
  //

  int arg_idx = 0;
  
  // sequencer
  KernelInfo sequencer = platform.find_kernel("sequencer");
  status = clSetKernelArg(sequencer.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"sequencer_kernel: Failed set arg %d.", arg_idx - 1);

  // retriever
  KernelInfo feature_reader = platform.find_kernel("retriever");

  arg_idx = 0;
  
  status = clSetKernelArg(feature_reader.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"feature_reader_kernel: Failed set arg %d.", arg_idx - 1);

  status = clSetKernelArg(feature_reader.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.wait_after_conv_cycles));
  checkError(status,"feature_reader_kernel: Failed set arg %d.", arg_idx - 1);
  
  // filter reader
  KernelInfo filter_reader = platform.find_kernel("filter_reader");
  
  arg_idx = 0;

  status = clSetKernelArg(filter_reader.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"filter_reader_kernel: Failed set arg %d.", arg_idx - 1);

  status = clSetKernelArg(filter_reader.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.filter_buffer));
  checkError(status,"filter_reader_kernel: Failed set arg %d.", arg_idx - 1);

  status = clSetKernelArg(filter_reader.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.bias_bn_buffer));
  checkError(status,"filter_reader_kernel: Failed set arg %d.", arg_idx - 1);

  // relu
  KernelInfo relu = platform.find_kernel("relu");

  arg_idx = 0;

  status = clSetKernelArg(relu.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"relu_kernel: Failed set arg %d.", arg_idx - 1);

  // pool
  KernelInfo pool = platform.find_kernel("pool");
  
  arg_idx = 0;

  status = clSetKernelArg(pool.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"pool_kernel: Failed set arg %d.", arg_idx - 1);

  // pool_tail
  KernelInfo pool_tail = platform.find_kernel("pool_tail");
  
  arg_idx = 0;

  status = clSetKernelArg(pool_tail.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"pool_tail_kernel: Failed set arg %d.", arg_idx - 1);

  status = clSetKernelArg(pool_tail.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.feature_ddr));
  checkError(status,"pool_tail_kernel: Failed set arg %d.", arg_idx - 1);

  // full_size_pool
  KernelInfo full_size_pool = platform.find_kernel("full_size_pool");

  arg_idx = 0;

  status = clSetKernelArg(full_size_pool.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"end_pool_kernel: Failed set arg %d.", arg_idx - 1);

  // feature_writer
  KernelInfo feature_writer = platform.find_kernel("feature_writer");
 
  arg_idx = 0;

  status = clSetKernelArg(feature_writer.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status,"feature_writer_kernel: Failed set arg %d.", arg_idx - 1);
  status = clSetKernelArg(feature_writer.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.feature_ddr));
  checkError(status,"feature_writer_kernel: Failed set arg %d.", arg_idx - 1);
  status = clSetKernelArg(feature_writer.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.bias_bn_buffer));
  checkError(status,"feature_writer: Failed set arg %d.", arg_idx - 1);

  // enqueue kernels
  EnqueueKernels(/* create_events */ false, /* enqueue_infinite_loop */ true);
  int torank[2] = {0};

  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  int HXW = H * W;

  const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;
 
  const int feature_ddr_size = OUTPUT_OFFSET + num_images * OUTPUT_OFFSET;
  float *input_raw_images = (float*)malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);
 
  for (int i = 0; i < num_images; i++) {
    // LoadInputImage(image_file, network.input_raw + i * C * HXW, input_raw_images, 0);
#ifdef IMAGENET
    LoadInputJpeg(image_file, network.input_raw + i * C * HXW, input_raw_images, 0);
#else
    // LoadInputJpeg(image_file, network.input_raw + i * C * HXW, input_raw_images, 0);
    LoadInputImage(image_file, network.input_raw + i * C * HXW, input_raw_images, 0);
#endif

    //conv simluation in host
    for (int sim_layer = 0; sim_layer < DEVICE_START_LAYER; sim_layer++){
      SimCNNLayer<int8,float>(sim_layer, network.filter_raw, network.bias_bn, network.input_raw, network.scale, network.sim_out);
      if(sim_layer < DEVICE_START_LAYER - 1){
        memset(network.input_raw, 0, sizeof(float) * network.input_raw_size);
        std::copy(network.sim_out,network.sim_out + network.input_raw_size, network.input_raw);
        memset(network.sim_out, 0, sizeof(float) * network.input_raw_size);
      }
    }

  }

  // InputConvert(network.input_raw, network.input, num_images);
  InputConvert(network.sim_out, network.input, num_images);

  // float trans = network.q[0] > 0 ? (1.0f / ( 1 << network.q[0])) : (1 << (-network.q[0]));
  float trans = 32.0; //first layer scale value
  for (int i = 0; i < input_device_size; i++) {
    #ifndef FBIT_MASK
    float tmp = network.input[i] * trans;
    #else
    float tmp = network.input[i];
    #endif
    int tmp_int = (int)(tmp > 0 ? tmp + 0.5 : tmp - 0.5);
    network.input_real[i] = tmp_int > REALFBITMAX ? REALFBITMAX : tmp_int < REALFBITMIN ? REALFBITMIN : tmp_int;
    #ifdef PRINT_H_INPUT_REAL
    printf("Input Real i=%d network.input[%d]=%f trans=%f tmp=%f input_real=%d\n", i, i, network.input[i], trans, tmp, network.input_real[i]);
    #endif
  }

  free(input_raw_images);

 #ifndef CHECK_H_SIMULATION_START_OUTPUT
  status = clEnqueueWriteBuffer(platform.input_queue, network.input_buffer[0], CL_TRUE, 0, sizeof(real)* input_device_size, network.input_real, 0, NULL, NULL);
  checkError(status, "Failed clEnqueueWriteBuffer : input_queue");
  clFinish(platform.input_queue);

  arg_idx = 0;
  KernelInfo input_reader = platform.find_kernel("input_reader");
  status = clSetKernelArg(input_reader.kernel, arg_idx++, sizeof(cl_int), (void*)&num_images);
  checkError(status, "input_reader_kernel: Failed set arg %d.", arg_idx - 1);

  status = clSetKernelArg(input_reader.kernel, arg_idx++, sizeof(cl_mem), (void*)&(network.input_buffer[0]));
  checkError(status, "input_reader_kernel: Failed set arg %d.", arg_idx - 1);

  INFO("Enqueuing Tasks...\n");
  EnqueueKernels(/* create_events */ true, /* enqueue_infinite_loop */ false);

  INFO("Starting...\n");

  WaitForAllKernels();

  INFO("Wait for finish...\n");

  cl_ulong total_sequencer = getStartEndTime(platform.find_kernel("sequencer").event);
  INFO("Latency = %0.3f ms\n", double(total_sequencer) * 1e-6);
  INFO("Throughput = %.1f fps\n", 1 / double(total_sequencer) / 1e-9 * num_images);

  // release events
  ReleaseAllEvents();

  // read output
  clEnqueueReadBuffer(platform.output_queue, network.feature_ddr, CL_TRUE, 0, sizeof(real) * feature_ddr_size, network.output, 0, NULL, NULL);
  checkError(status,"Failed clEnqueueReadBuffer : output_queue");
  clFinish(platform.output_queue);

  #ifdef FBIT_SIMULATION_END_OUTPUT
  OutputRConvert(network.output, network.output_pre, num_images);

  memset(network.sim_out, 0, sizeof(float) * network.input_raw_size);

  for (int sim_layer = DEVICE_END_LAYER; sim_layer < NUM_CONVOLUTIONS; sim_layer++){
      SimCNNLayer<int8,float>(sim_layer, network.filter_raw, network.bias_bn, network.output_pre, network.scale, network.sim_out);
      if(sim_layer > DEVICE_END_LAYER){
        memset(network.output_pre, 0, sizeof(float) * network.input_raw_size);
        std::copy(network.sim_out,network.sim_out + network.input_raw_size, network.output_pre);
        memset(network.sim_out, 0, sizeof(float) * network.input_raw_size);
      }
  }
  #endif

  #endif
}

