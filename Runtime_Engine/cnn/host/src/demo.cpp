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
#include "demo.h"

template void Demo::Softmax(int n, float (*scale)[2], float *output);

Demo::Demo() {

}

bool Demo::Init() {
 this->top1 = 0;
 this->top5 = 0;
 this->top1score = 0;
 this->top5score = 0;
 memset(this->top_labels, -1, sizeof(this->top_labels));
 if(!LoadLabel_Demo())
 return -1;

 return true;
}

void Demo::CleanUp() {
  this->imagecontents.clear();
  this->imagenet_labels.clear();
}

bool Demo::LoadLabel_Demo() {
  // std::ifstream fin("../imagenet_test_images/imagenet1000_clsid_to_human.txt");
  std::ifstream fin("../imagenet1000_clsid_to_human.txt");
  std::string line;
  int position, pos_s_m_start, pos_d_m_start, pos_m_end, pos_q_end;
  
  if(fin) {
    while(getline(fin,line)) {
      imagenet_content image_item;
      position = line.find(": ");
      
      std::string imgnet_mid_arr_index = line.substr(0,position);

      std::string imgnet_mid_arr_content_mid = line.substr(position+2,line.size()-1);
        
      std::string imgnet_mid_arr_content_q, imgnet_mid_arr_content;
            
      pos_d_m_start = imgnet_mid_arr_content_mid.find_first_of("\"");
    
      if(pos_d_m_start != -1) {
        pos_m_end = imgnet_mid_arr_content_mid.find_last_of("\"");
        imgnet_mid_arr_content_q = imgnet_mid_arr_content_mid.substr(pos_d_m_start+1,pos_m_end-1);
      } else {
        pos_s_m_start = imgnet_mid_arr_content_mid.find_first_of("\'");
        pos_m_end = imgnet_mid_arr_content_mid.find_last_of("\'");
        imgnet_mid_arr_content_q = imgnet_mid_arr_content_mid.substr(pos_s_m_start+1,pos_m_end-1);
      }	

      pos_q_end = imgnet_mid_arr_content_q.find_first_of(",");

      if( pos_q_end != -1) {
        imgnet_mid_arr_content = imgnet_mid_arr_content_q.substr(0,pos_q_end);
      } else {
        imgnet_mid_arr_content = imgnet_mid_arr_content_q;
      }

      image_item.index = atoi(imgnet_mid_arr_index.c_str());
      image_item.label_name = imgnet_mid_arr_content.c_str();          
      this->imagecontents.push_back(image_item);
    }
  } else {
    /* code */
    std::cout << "fail to open file imagenet1000.txt" << std::endl;
    return -1;
  }
  
  fin.close();

  // std::ifstream imagenet_img("../imagenet_test_images/val_map.txt");
  std::ifstream imagenet_img("../val_map.txt");
  std::string line_addr_imagenet_img;
  int position_imagenet,  pos_s_m_start_imagenet, pos_d_m_start_imagenet, pos_m_end_imagenet, pos_q_end_imagenet;
  //getline(imagenet_img,line_addr_imagenet_img);
  if(imagenet_img) {
    while(getline(imagenet_img,line_addr_imagenet_img)) {
      imagenet_label image_label;
      position_imagenet = line_addr_imagenet_img.find("\t");
      std::string jpg_image_name = line_addr_imagenet_img.substr(0,position_imagenet);
      image_label.jpg_image_name = jpg_image_name.c_str();
      int label_index = atoi(line_addr_imagenet_img.substr(position_imagenet+1,line_addr_imagenet_img.size()-1).c_str());
      image_label.label_index = label_index;
      this->imagenet_labels.push_back(image_label);
    }
  } else {
    /* code */
    std::cout << "fail to open file val_map.txt" << std::endl;
    return -1;
  }

  imagenet_img.close();
  
  return true;
}

template<typename T>
void Demo::Softmax(int n, float (*scale)[2], T* output) {
  // int output_channel = kOutputChannels[NUM_LAYER - 1];
  // #ifdef SQUEEZENET
  // int width = kPoolOutputWidth[NUM_LAYER - 1];
  // int height = kPoolOutputHeight[NUM_LAYER - 1];
  // #else
  // int width = 1;
  // int height = 1;
  // #endif
  
  // int size = output_channel * width * height;

  // FILE *fp;
  // fp = fopen("Lastconv.dat","wt");
  // int H = height;
  // int W = width;
  // int ddr_write_offset = kDDRWriteBase[NUM_LAYER - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
  // int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  // int pad = 0;

  // float total_error = 0.;
  // float total_expect = 0.;

  // std::vector<StatItem> stat_array;

  // // float sum_exp = 0;
  // double sum_exp = 0;
  // #ifdef SQUEEZENET
  // float max_ = 0.0;
  // float average_pool_value[output_channel];
  // memset(average_pool_value, 0.0, output_channel*sizeof(float));
  // #endif
  

  // for (int n = 0; n < output_channel; n++) {
  //   for (int h = 0; h < H; h++) {
  //     for (int w = 0; w < W; w++) {
  //         int n_vec = n / N_VECTOR;
  //         // int h_vec = 0;
  //         // int w = 0;
  //         int h_vec = h;
  //         int w_vec = w / W_VECTOR;
  //         int ww = w - w_vec * W_VECTOR;
  //         int nn = n - n_vec * N_VECTOR;
  //         int addr_out = 
  //                     ddr_write_offset + output_offset +
  //                     n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
  //                     h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
  //                     w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
  //                     ww * NARROW_N_VECTOR +
  //                     nn;;
  //         int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
  //         // float trans = 1 << (-current_q); //take care of shortcut
  //         float trans = (current_q <=0) ? 1 << (-current_q) : (1.0/(1 << current_q)); //take care of shortcut
  //         #ifdef SQUEEZENET
  //         average_pool_value[n] += (output[addr_out]/trans/(H*W));
  //         if(average_pool_value[n] > max_)
  //         {
  //           max_ = average_pool_value[n];
  //         }
  //         #endif
  //     }
  //   }

  //   StatItem temp;
  //   temp.label = n;
  //   // temp.feature = output[addr_out] / trans;
  //   #ifdef SQUEEZENET
  //   // temp.feature = average_pool_value[n] - max_;
  //   temp.feature = average_pool_value[n];
  //   #else
  //   temp.feature = output[addr_out] / trans;
  //   #endif

  //   sum_exp += exp(temp.feature);

  //   // printf("channel is %d, sum_exp is %lf\n", n, sum_exp);

  //   stat_array.push_back(temp);
  // }

  // for (int i = 0; i < 5; i++) {
  //   for (int j = 0; j < output_channel - i - 1; j++) {
  //     if (stat_array[j].feature > stat_array[j + 1].feature) {
  //       std::swap(stat_array[j], stat_array[j + 1]);
  //     }
  //   }
  // }

  // for (int i = 0; i < 5; i++) {
  //   top_labels[i] = stat_array[ output_channel - i - 1 ].label;
  //   softmax_result[i] = exp(stat_array[output_channel - i - 1].feature) / sum_exp;
  //   INFO("rank=%d\tlabel=%5d\tprobability=%lf\n", i, top_labels[i], exp(stat_array[output_channel - i - 1].feature) / sum_exp);
  // }

  // fclose(fp);

#ifdef  FBIT_SIMULATION_END_OUTPUT
  int output_layer = NUM_CONVOLUTIONS;
#else
  int output_layer = NUM_LAYER;
#endif

  int output_channel = kOutputChannels[output_layer - 1];
  int width = 1;
  int height = 1;
  
  int size = output_channel * width * height;

  FILE *fp;
  fp = fopen("Lastconv.dat","wt");
  int H = height;
  int W = width;
  int ddr_write_offset = kDDRWriteBase[output_layer - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;

  std::vector<StatItem> stat_array;

  float sum_exp = 0;

  for (int n = 0; n < output_channel; n++) {
    int n_vec = n / N_VECTOR;
    int h_vec = 0;
    int h = 0;
    int w = 0;
    int w_vec = 0;
    int ww = w - w_vec * W_VECTOR;
    int nn = n - n_vec * N_VECTOR;
    #if (!defined(CHECK_H_SIMULATION_START_OUTPUT) && !defined(FBIT_SIMULATION_END_OUTPUT))
    int addr_out = 
                ddr_write_offset + output_offset +
                n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                ww * NARROW_N_VECTOR +
                nn;
    #endif
    // int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
    // float trans = 1 << (-current_q); //take care of shortcut
    #if (defined(CHECK_H_SIMULATION_START_OUTPUT) || defined(FBIT_SIMULATION_END_OUTPUT))
    int addr_exp = n * H * W + h * W + w;
    int addr_out = addr_exp;
    #endif
    float trans = scale[output_layer - 1][1];

    StatItem temp;
    temp.label = n;
    temp.feature = output[addr_out] / trans;

    sum_exp += exp(temp.feature);

    stat_array.push_back(temp);

    //printf("Evaluation n=%d, feature=%f, addr_out=%d, trans=%f sum_exp=%f\n", n, stat_array[n].feature, addr_out, trans, sum_exp);
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < output_channel - i - 1; j++) {
      if (stat_array[j].feature > stat_array[j + 1].feature) {
        std::swap(stat_array[j], stat_array[j + 1]);
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    top_labels[i] = stat_array[output_channel - i - 1].label;
    //INFO( "rank=%d\tlabel=%5d\tfeature=%f\n", i, top_labels[i], stat_array[ output_channel - i - 1 ].feature );
    INFO("rank=%d\tlabel=%5d\tprobability=%f\n", i, top_labels[i], exp(stat_array[output_channel - i - 1].feature) / sum_exp);
  }

  fclose(fp);
}

void Demo::Evaluation(int test_index) {
  label = imagenet_labels[test_index].label_index;
  top1score = top_labels[0] == label;

  top5score = 0;
  for( int i = 0; i < 5; i++ ) {
    if( top_labels[i] == label )
      top5score = 1;
  }
}

void Demo::Result(cl_ulong total_sequencer, int num_images, double total_time) {
  int index_top5;
  FILE *fp_info = fopen("/tmp/fpga.info","w+");
  if(fp_info == NULL)
  printf("Open fpga.info file error.\n");
  double time_d_s = double(total_sequencer);
  float throughput = 1 * num_images / time_d_s / 1e-9;
  float efficiency_tmp = throughput/45;
  fprintf(fp_info, "latency:%.3f\n", total_time);//?
  printf("latency:%.3f\n", total_time);//?
  fprintf(fp_info, "throughput:%.1f\n", throughput);
  printf("throughput:%.1f\n", 1 * num_images / time_d_s / 1e-9);
  printf("total_sequencer value:%.5f\n",time_d_s);
  fprintf(fp_info, "efficiency:%.3f\n", efficiency_tmp);

  for(int i = 0; i < 5; i++) {
      index_top5 = top_labels[i];
      printf("index_top5 is %d\n",index_top5);
      if(i==0) fprintf(fp_info, "result:%s\n", imagecontents[index_top5].label_name.c_str());
      if(i==0) printf("result:%s\n", imagecontents[index_top5].label_name.c_str());
      fprintf(fp_info, "top:%s#%.3f%%\n", imagecontents[index_top5].label_name.c_str(), softmax_result[i]*100);
      printf("top%d:%s %.3f%%\n", i+1, imagecontents[index_top5].label_name.c_str(), softmax_result[i]*100);
  }

  fclose(fp_info);
}
