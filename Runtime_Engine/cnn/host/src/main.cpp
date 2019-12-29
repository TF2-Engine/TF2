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
#include "demo.h"

#define IMGNETITEM_TEST_NUM 50000

int main(int argc, char **argv) {
  if (argc != 6) {
    INFO("USAGE:\n%s <model_file> <quantization_file> <image_file> <verify_file> <num_images>\n", argv[0]);
    return 1;
  }

  char *model_file = argv[1];
  char *q_file = argv[2];
  char *image_file = argv[3];
  char *verify_file_name = argv[4];
  int num_images = atoi(argv[5]);

  INFO("model_file = %s\n", model_file);
  INFO("q_file = %s\n", q_file);
  INFO("image_file  = %s\n", image_file);
  INFO("verify_file_name = %s\n", verify_file_name);
  INFO("num_images = %d\n", num_images);

  OpenCLFPGA platform;
  if (!platform.Init()) {
    return -1;
  }  

  Demo demo;
  if (!demo.Init())
  {
    return -1;
  }

  NetWork network;
  if (!network.Init(platform, model_file, q_file, image_file, num_images)) {
    return -1;
  }

  Runner runner(platform, network);
  runner.Init();
  //Runner.Run();

  for(int test_index=0; test_index< IMGNETITEM_TEST_NUM; test_index++)
  {
    std::string line_addr_img = "../imagenet_test_images/"+demo.imagenet_labels[test_index].jpg_image_name;
    std::ifstream fin_img_addr;
    fin_img_addr.open(const_cast<char*>(line_addr_img.c_str()));

    if(fin_img_addr)
    {
      char* image_file = const_cast<char*>(line_addr_img.c_str());

      runner.Run(image_file);

      demo.Softmax(runner.num_images - 1, network.q, network.output);
      Evaluation(runner.num_images - 1, network.q, network.output, network.top_labels); 

      runner.end_time = getCurrentTimestamp();
      const double total_time = runner.end_time - runner.start_time;

      demo.Result(runner.total_sequencer, runner.num_images, total_time);

      //std:: remove(const_cast<char*>(line_addr_img.c_str()));
      fin_img_addr.close();

      demo.Evaluation(test_index);

      demo.top1 +=demo.top1score;
      demo.top5 +=demo.top5score;

      printf("index is %d, imagenet_orl_label is %d, top1score is %d, top5score is %d, top1 is %d, top5 is %d\n", test_index, demo.imagenet_labels[test_index].label_index, demo.top1score, demo.top5score, demo.top1, demo.top5);
    }
  }

  float accuracy_top1 = (1.0*demo.top1)/IMGNETITEM_TEST_NUM;
  float accuracy_top5 = (1.0*demo.top5)/IMGNETITEM_TEST_NUM;

  printf("top1 accuracy is %.3f\n", accuracy_top1);
  printf("top5 accuracy is %.3f\n", accuracy_top5);

  // verification
  /*
  for (int i = 0; i < num_images; i++) {
    Verify(i, verify_file_name, network.q, network.output);
    Evaluation(i, network.q, network.output, network.top_labels);
  }*/

  demo.CleanUp();

  // CleanUp
  network.CleanUp();
  platform.CleanUp();

  return 0;
}
