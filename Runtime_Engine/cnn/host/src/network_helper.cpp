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

#include "network_helper.h"

void evaluation( int n, char* q, real* output, int* top_labels )
{
  int output_channel = output_channels[NUM_LAYER - 1];
  int width = 1;
  int height = 1;
  
  int size = output_channel * width * height;

  FILE *op;
  op = fopen("Lastconv.dat","wt");
  int H = height;
  int W = width;
  int ddr_write_offset = ddr_write_base[ NUM_LAYER - 1 ] * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR);
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;

  std::vector<stat_item> stat_array;

  float sum_exp = 0;

  for(int k = 0; k < output_channel; k++){
    int kvec = k / K_VECTOR;
    int hvec = 0;
    int w = 0;
    int wvec = 0;
    int ww = w - wvec * W_VECTOR;
    int kk = k - kvec * K_VECTOR;
    int addr_out = 
                ddr_write_offset + output_offset +
                kvec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR) +
                hvec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR) +
                wvec * NEXT_POWER_OF_2(W_VECTOR * RELU_K_VECTOR) +
                ww * RELU_K_VECTOR +
                kk;;
    int current_q = q[ NUM_LAYER * MAX_OUT_CHANNEL + k ];
    float trans = 1 << ( -current_q ); //take care of shortcut

    stat_item temp;
    temp.label = k;
    temp.feature = output[addr_out] / trans;

    sum_exp += exp( temp.feature );

    stat_array.push_back( temp );
  }

  for( int i = 0; i < 5; i++ ) {
    for( int j = 0; j < output_channel - i - 1; j++ ) {
      if( stat_array[j].feature > stat_array[ j + 1 ].feature ) {
        std::swap( stat_array[j], stat_array[ j + 1 ] );
      }
    }
  }

  //for( int i = 0; i < output_channel; i++ )
    //std::cout << i << stat_array[i].label << stat_array[i].feature << std::endl;

  for( int i = 0; i < 5; i++ ) {
    top_labels[i] = stat_array[ output_channel - i - 1 ].label;
    //INFO( "rank=%d\tlabel=%5d\tfeature=%f\n", i, top_labels[i], stat_array[ output_channel - i - 1 ].feature );
    INFO( "rank=%d\tlabel=%5d\tprobability=%f\n", i, top_labels[i], exp( stat_array[ output_channel - i - 1 ].feature ) / sum_exp );
  }

  fclose(op);
}

void Load_label(int Num,int *labels)
{
  FILE *fp;
  if((fp = fopen("label.dat","r")) == NULL){
    ERROR("Error in search label.dat\n");
    exit(0);
  }

  for(int i = 0; i < Num; i++) {
    fscanf(fp,"%d",&labels[i]);
  }

  fclose(fp);
}
