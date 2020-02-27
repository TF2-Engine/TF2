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

// #include "cnn.h"
// #include "CL/opencl.h"
// #include "ACLHostUtils.h"
// #include "AOCL_Utils.h"
#include "fpganetworkinterface.h"
#include "cycles_computation.h"
#include "tf2_auto_param.h"
#include <ctype.h>
#include <algorithm>
#include <cmath>
#include <cstring>

std::vector<std::vector<int> >framework;

module_framework_data FrameParseStore(std::string filename, std::string netname) {
    
   module_framework_data module_framework;

   memset(&(module_framework.total_framework), 0, sizeof(a_frame));

   StFpgaNetInfo *pstNetInfo;
   pstNetInfo=(StFpgaNetInfo*)malloc(Max_Param_size*sizeof(char));
   read_fpganetwork(filename,pstNetInfo);

   int size_memory=0;

   void *pvMemory=(void*)pstNetInfo;

   int last_input_layer_id;
 
   int num_block = pstNetInfo->iLayerNum;

   size_memory+=sizeof(StFpgaNetInfo);
   
   conv_info_block conv_block[num_block];
   int id_conv[num_block];
   bool is_concate[num_block];
   bool is_inherit_concate[num_block];
   std::vector<int> conv_siblings[num_block];
   block_sibling block_siblings[num_block]; //summarize the inputblock's output
   int input_id = 0;

   memset(conv_block, 0, sizeof(conv_info_block)*num_block);
   memset(is_concate, 0, sizeof(conv_info_block)*num_block);
   memset(is_inherit_concate, 0, sizeof(conv_info_block)*num_block);
   memset(id_conv, 0, sizeof(int) * num_block);
   memset(block_siblings, 0, sizeof(block_sibling) * num_block);

   int conv_count = -1;
   int index_concate = -1;
   int input_index_base = 0;

   int layer_num = 0;
   int conv_input_count = 0;

   int max_cache_read_base = (!strcmp(netname.c_str(),"googlenet")) ? 3 : 2;

   int current_conv_output_channel = 0;
   int current_conv_input_channel = 0;
   int current_conv_output_height = 0;
   int current_conv_output_width = 0;
   int current_conv_input_height = 0;
   int current_conv_input_width = 0;
   int current_filter = 0;

   int current_pool_output_channel = 0;
   int current_pool_input_channel = 0;
   int current_pool_output_height = 0;
   int current_pool_output_width = 0;
   int current_pool_input_height = 0;
   int current_pool_input_width = 0;
   int current_pool_window_size = 0;

   int current_pool_pad_size = 0;

   int current_window_5_input_channel = 0;
   int current_window_3_input_channel = 0;
   int current_window_1_input_channel = 0;

   int common_buffer_num = 2;
   int last_cache_num = 0;
   int cache_num_max = common_buffer_num;
   int reset_cache_num = 1;
   int reset_input_cache_num = 0;
   int last_sub_cache_num = 0;
   bool multi_write_cache_flag = false;
   int last_ddr_read_num = 2;
   int last_ddr_write_num = 2;
   int common_read_ddr_num = 0;
   int common_write_ddr_num = 0;

   for(int i = 0; i < num_block; i++)
   {
		StFpgaLayerInfo *pstLayerInfo=(StFpgaLayerInfo*)((char*)pvMemory+size_memory);
		size_memory+=sizeof(StFpgaLayerInfo);
      pstNetInfo->pstFpgaLayerInfo=pstLayerInfo;

      id_conv[i] = -1; //initializaiton
      conv_block[i].pool_max = 1;
      input_id = pstLayerInfo->ariInputBlobId[0];

      if(input_id != -1) 
      {
         block_siblings[input_id].id.push_back(i);
         for(int iblob = 0; iblob < pstLayerInfo->iBlobOutputNum; iblob++) //only once
         {
            block_siblings[input_id].N = pstLayerInfo->arstOutputfeatDim[iblob].C;//N
            block_siblings[input_id].H = pstLayerInfo->arstOutputfeatDim[iblob].H;//H+
            block_siblings[input_id].W = pstLayerInfo->arstOutputfeatDim[iblob].W;//W+
         }

         if(pstLayerInfo->iBlobInputNum > 1) 
         {
            cache_num_max = common_buffer_num;
            reset_cache_num = (!strcmp(netname.c_str(),"googlenet")) ? (3 - reset_input_cache_num) : 1;

            multi_write_cache_flag = false;
         }
      }
      
      for(int iop = 0; iop < pstLayerInfo->iOpNum; iop++)
      {
         StFpgaOpInfo *pstOpInfo = (StFpgaOpInfo*)((char*)pvMemory+size_memory);
         pstLayerInfo->pstFpgaOpInfo=pstOpInfo;
         size_memory+=sizeof(StFpgaOpInfo);

         if(pstOpInfo->enumOpType==OpConv)
         {  
            conv_count++;
            if(input_id != -1)  //for cache_read_base
            {
               if(block_siblings[input_id].id.size() == 1)
               {
                  if(pstLayerInfo->iBlobInputNum > 1) cache_num_max = common_buffer_num;
                  block_siblings[input_id].cache_read_base_num = last_cache_num = (last_cache_num < cache_num_max) ? ++last_cache_num : reset_cache_num;
               }
               else if (block_siblings[input_id].id.size() > 1)
               {
                  cache_num_max = max_cache_read_base;
                  reset_input_cache_num = block_siblings[input_id].cache_read_base_num;
                  last_cache_num = (!strcmp(netname.c_str(),"googlenet")) ? 2 : last_cache_num;

                  multi_write_cache_flag = true;
               }
            }
            else
            {
               /* code */
               ++last_cache_num;
            }
            
            id_conv[i] = conv_count;
    
            if(conv_count == 0)// get first layer num and block ID
            {
               std::vector<int> id_info_f;

               id_info_f.push_back(pstLayerInfo->iLayerId);
               framework.push_back(id_info_f);
               layer_num ++;
               last_input_layer_id = pstLayerInfo->ariInputBlobId[0];
            }
            if(last_input_layer_id != pstLayerInfo->ariInputBlobId[0])
            {
               layer_num ++;
               last_input_layer_id = pstLayerInfo->ariInputBlobId[0];
            }

            if(conv_count >= 1) module_framework.conv_blocks.push_back(conv_block[conv_count - 1]);//push back last block
            module_framework.total_framework.num_conv =  conv_count;

            pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
            size_memory+=sizeof(StConvParam);
            StConvParam *pstNetConvparam = (StConvParam *)pstOpInfo->pvOpInfo;

            conv_block[conv_count].basic_info_frame.enable_info[0] = true;
            conv_block[conv_count].basic_info_frame.specific_struct_enable[5] = pstNetConvparam->bias_term; //for bias enable info
            //get N,C,H,W
            for(int iblob = 0; iblob < pstLayerInfo->iBlobInputNum; iblob++)
            {
               if(conv_count == 0)
               {
                  module_framework.total_framework.input_image_channel = (!strcmp(netname.c_str(),"googlenet")) ? 3 : pstLayerInfo->arstInputfeatDim[iblob].C;
                  module_framework.total_framework.input_image_height = pstLayerInfo->arstInputfeatDim[iblob].H;
                  module_framework.total_framework.input_image_width = pstLayerInfo->arstInputfeatDim[iblob].W;
                  module_framework.total_framework.first_filter_size = pstNetConvparam->ikernel_h;
               }

               if(((strcmp(netname.c_str(),"resnet50") || strcmp(netname.c_str(),"googlenet"))) && conv_count == 0)
               {
                  conv_block[conv_count].size_block[0].input_feature_size[0] = 27; //C
                  conv_block[conv_count].size_block[0].input_feature_size[1] = 114; //H
                  conv_block[conv_count].size_block[0].input_feature_size[2] = 114; //W
               }
               else
               {
                  conv_block[conv_count].size_block[0].input_feature_size[0] = pstLayerInfo->arstInputfeatDim[iblob].C;//C
                  conv_block[conv_count].size_block[0].input_feature_size[1] = pstLayerInfo->arstInputfeatDim[iblob].H;
                  conv_block[conv_count].size_block[0].input_feature_size[2] = pstLayerInfo->arstInputfeatDim[iblob].W;
               }

            }

            for(int iblob = 0; iblob < pstLayerInfo->iBlobOutputNum; iblob++)
            {
               conv_block[conv_count].size_block[0].output_feature_size[0] = pstLayerInfo->arstOutputfeatDim[iblob].C;//N
               conv_block[conv_count].size_block[0].output_feature_size[1] = pstLayerInfo->arstOutputfeatDim[iblob].H;//H+
               conv_block[conv_count].size_block[0].output_feature_size[2] = pstLayerInfo->arstOutputfeatDim[iblob].W;//W+

               conv_block[conv_count].size_block[1].input_feature_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//C
               conv_block[conv_count].size_block[1].input_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];
               conv_block[conv_count].size_block[1].input_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];

               conv_block[conv_count].size_block[1].output_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];//H+ for pool initialization
               conv_block[conv_count].size_block[1].output_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];//W+
               
            }

            conv_block[conv_count].size_block[0].window_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//N
            conv_block[conv_count].size_block[0].window_size[1] = conv_block[conv_count].size_block[0].input_feature_size[0];//C

            conv_block[conv_count].size_block[0].window_size[2] = (pstNetConvparam->ikernel_h == 7) ? 3 : pstNetConvparam->ikernel_h;
            conv_block[conv_count].size_block[0].window_size[3] = (pstNetConvparam->ikernel_w == 7) ? 3 : pstNetConvparam->ikernel_w;

            if(((!strcmp(netname.c_str(),"resnet50") || !strcmp(netname.c_str(),"googlenet"))) && conv_count == 0)
            {
               conv_block[conv_count].size_block[0].stride_size = 1;
               conv_block[conv_count].size_block[0].pad_size = 0;
            }
            else
            {
               conv_block[conv_count].size_block[0].pad_size = pstNetConvparam->ipad_left;
               conv_block[conv_count].size_block[0].stride_size = pstNetConvparam->istride_h;
            }

            conv_block[conv_count].basic_info_frame.base_num[0] = (input_id == -1) ? 1 : block_siblings[input_id].cache_read_base_num; //add cachereadbase value
            conv_block[conv_count].basic_info_frame.concate_offset[1] = conv_block[conv_count].size_block[0].output_feature_size[0]; //initialize the kNEnd value

            if(input_id != -1) //for cache_write_base
            {
               if(block_siblings[input_id].id.size() == 1)
               {
                  if((!strcmp(netname.c_str(),"googlenet")))
                  {
                     if(!multi_write_cache_flag) last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                     else last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - conv_block[conv_count - 1].basic_info_frame.base_num[1];
                  }
                  else
                  {
                     /* code */
                     last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                  }
                  
                  
                  conv_block[conv_count].index_input_layer = conv_count;
               }
               else if(block_siblings[input_id].id.size() > 1)
               {
                  if((!strcmp(netname.c_str(),"googlenet")))
                  conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - last_sub_cache_num;
                  else
                  last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                  
                  
                  int id_first_sibling = block_siblings[input_id].id[0];
                  int conv_first_sibling = id_conv[id_first_sibling];
                  if(!is_concate[input_id] && !is_inherit_concate[input_id])
                  conv_block[conv_count].index_input_layer = id_conv[id_first_sibling];
                  else
                  {
                    conv_block[conv_first_sibling].input_concate_connection = conv_block[conv_count].input_concate_connection = true;
                    module_framework.conv_blocks[conv_first_sibling].input_concate_connection = conv_block[conv_first_sibling].input_concate_connection;
                  }

               }
            }
            else
            {
                 conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];

                 conv_block[conv_count].index_input_layer = conv_count;
            }

            conv_block[conv_count].basic_info_frame.mem_write_enable[0] = 1; //cache write enable
            //get max value
            current_conv_output_channel = conv_block[conv_count].size_block[0].output_feature_size[0];
            module_framework.total_framework.max_output_channel = (module_framework.total_framework.max_output_channel > current_conv_output_channel) \
            ? module_framework.total_framework.max_output_channel : current_conv_output_channel;

            current_conv_input_channel = conv_block[conv_count].size_block[0].input_feature_size[0];
            module_framework.total_framework.max_input_channel = (module_framework.total_framework.max_input_channel > current_conv_input_channel) \
            ? module_framework.total_framework.max_input_channel : current_conv_input_channel;

            current_conv_input_height = conv_block[conv_count].size_block[0].input_feature_size[1];
            module_framework.total_framework.max_input_height = (module_framework.total_framework.max_input_height > current_conv_input_height) \
            ? module_framework.total_framework.max_input_height : current_conv_input_height;

            current_conv_input_width = conv_block[conv_count].size_block[0].input_feature_size[2];
            module_framework.total_framework.max_input_width = (module_framework.total_framework.max_input_width > current_conv_input_width) \
            ? module_framework.total_framework.max_input_width : current_conv_input_width;

            current_conv_output_height = conv_block[conv_count].size_block[0].output_feature_size[1];
            module_framework.total_framework.max_output_height = (module_framework.total_framework.max_output_height > current_conv_output_height) \
            ? module_framework.total_framework.max_output_height : current_conv_output_height;

            current_conv_output_width = conv_block[conv_count].size_block[0].output_feature_size[2];
            module_framework.total_framework.max_output_width = (module_framework.total_framework.max_output_width > current_conv_output_width) \
            ? module_framework.total_framework.max_output_width : current_conv_output_width;

            current_filter = conv_block[conv_count].size_block[0].window_size[2];
            module_framework.total_framework.max_filter = (module_framework.total_framework.max_filter > current_filter) \
            ? module_framework.total_framework.max_filter : current_filter;

            if(current_filter == 3) current_window_3_input_channel = current_conv_input_channel;
            if(current_filter == 1) current_window_1_input_channel = current_conv_input_channel;

            module_framework.total_framework.max_filter_size_1 = (module_framework.total_framework.max_filter_size_1  > current_window_1_input_channel) \
            ? module_framework.total_framework.max_filter_size_1 : current_window_1_input_channel;

            module_framework.total_framework.max_filter_size_2 = (module_framework.total_framework.max_filter_size_2  > current_window_3_input_channel) \
            ? module_framework.total_framework.max_filter_size_2 : current_window_3_input_channel;

            module_framework.total_framework.max_bias_size = module_framework.total_framework.max_output_channel;

         }
         if(pstOpInfo->enumOpType==OpBn)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StBnParam);
				StBnParam *pstNetBnparam;
				pstNetBnparam=(StBnParam*)(pstOpInfo->pvOpInfo);
            conv_block[conv_count].basic_info_frame.enable_info[3] = true;
				
			}
			if(pstOpInfo->enumOpType==OpScale)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StScaleParam);
				
			}

         if(pstOpInfo->enumOpType == OpPool)
         {
            pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StPoolParam);
				StPoolParam *pstNetPoolparam;
				pstNetPoolparam=(StPoolParam*)(pstOpInfo->pvOpInfo);

            int conv_pool_count;

            conv_block[conv_count].basic_info_frame.mem_write_enable[0] = (!strcmp(netname.c_str(),"googlenet")) ?  1 : ((pstNetPoolparam->enPoolMethod == PoolAvg) ? 0 : 1); //cache write enable
            conv_block[conv_count].basic_info_frame.specific_struct_enable[0] = (pstNetPoolparam->enPoolMethod == PoolAvg) ? 1 : 0; //cache write enable
            conv_block[conv_count].pool_max = (pstNetPoolparam->enPoolMethod == PoolAvg) ? 0 : 1;
            //if(googlenet){}
            if(block_siblings[input_id].id.size()>1) //add independent pooling
            {
               conv_count++;

               id_conv[i] = conv_count; //record for independent pooling

               if(conv_count >= 1) module_framework.conv_blocks.push_back(conv_block[conv_count - 1]);//push back last block
 
               //StFcParam
               conv_block[conv_count].basic_info_frame.specific_struct_enable[4] = 1;//independent pooling enable
               conv_block[conv_count].basic_info_frame.mem_write_enable[0] = 1; //cache write enable

               if(input_id != -1) 
               {
                  if(block_siblings[input_id].id.size() == 1)
                  {
                     if(pstLayerInfo->iBlobInputNum > 1) cache_num_max = common_buffer_num;
                     block_siblings[input_id].cache_read_base_num = last_cache_num = (last_cache_num < cache_num_max) ? ++last_cache_num : reset_cache_num;
                  }
                  else if (block_siblings[input_id].id.size() > 1)
                  {
                     cache_num_max = max_cache_read_base;
                     reset_input_cache_num = block_siblings[input_id].cache_read_base_num;
                     last_cache_num = 2; //reserve the buffer2
                  }
               }
               else
               {
                  /* code */
                  ++last_cache_num;
               }
               
               // conv_process_flag = true;
               conv_block[conv_count].size_block[0].input_feature_size[0] = 1;//C
               conv_block[conv_count].size_block[0].input_feature_size[1] = block_siblings[input_id].H;
               conv_block[conv_count].size_block[0].input_feature_size[2] = block_siblings[input_id].W;

               conv_block[conv_count].size_block[0].output_feature_size[0] = block_siblings[input_id].N;//N
               conv_block[conv_count].size_block[0].output_feature_size[1] = block_siblings[input_id].H;//H+
               conv_block[conv_count].size_block[0].output_feature_size[2] = block_siblings[input_id].W;//W+


               conv_block[conv_count].size_block[1].input_feature_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//C for pool input
               conv_block[conv_count].size_block[1].input_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];
               conv_block[conv_count].size_block[1].input_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];

               conv_block[conv_count].size_block[0].window_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//N
               conv_block[conv_count].size_block[0].window_size[1] = conv_block[conv_count].size_block[0].input_feature_size[0];//C
               conv_block[conv_count].size_block[0].window_size[2] = 1;
               conv_block[conv_count].size_block[0].window_size[3] = 1;

               conv_block[conv_count].size_block[0].stride_size = 1;
               conv_block[conv_count].size_block[0].pad_size = 0;

               conv_block[conv_count].basic_info_frame.base_num[0] = (input_id == -1) ? 1 : block_siblings[input_id].cache_read_base_num; //add cachereadbase value
               conv_block[conv_count].basic_info_frame.concate_offset[1] = conv_block[conv_count].size_block[0].output_feature_size[0]; //initialize the kNEnd value

               if(input_id != -1) 
               {
                  if(block_siblings[input_id].id.size() == 1)
                  {
                     if((!strcmp(netname.c_str(),"googlenet")))
                     {
                        if(!multi_write_cache_flag) last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                        else last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - conv_block[conv_count - 1].basic_info_frame.base_num[1];
                     }
                     else
                     {
                        /* code */
                        last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                     }

                     conv_block[conv_count].index_input_layer = conv_count;

                  }
                  else if(block_siblings[input_id].id.size() > 1)
                  {
                     if((!strcmp(netname.c_str(),"googlenet")))
                        conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - last_sub_cache_num;
                     else
                     {
                        /* code */
                        last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                     }
                     
                     int id_first_sibling = block_siblings[input_id].id[0];
                     int conv_first_sibling = id_conv[id_first_sibling];
                     if(!is_concate[input_id] && !is_inherit_concate[input_id])
                     conv_block[conv_count].index_input_layer = id_conv[id_first_sibling];
                     else
                     {
                        is_inherit_concate[i] = false; //for current pool after concate situation
                        conv_block[conv_first_sibling].input_concate_connection = conv_block[conv_count].input_concate_connection = true;
                        module_framework.conv_blocks[conv_first_sibling].input_concate_connection = conv_block[conv_first_sibling].input_concate_connection;
                     }
                     
                  }
               }
               else
               {
                  conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                  conv_block[conv_count].index_input_layer = conv_count;
               }

               //get max value
               current_conv_output_channel = conv_block[conv_count].size_block[0].output_feature_size[0];
               module_framework.total_framework.max_output_channel = (module_framework.total_framework.max_output_channel > current_conv_output_channel) \
               ? module_framework.total_framework.max_output_channel : current_conv_output_channel;

               current_conv_input_channel = conv_block[conv_count].size_block[0].input_feature_size[0];
               module_framework.total_framework.max_input_channel = (module_framework.total_framework.max_input_channel > current_conv_input_channel) \
               ? module_framework.total_framework.max_input_channel : current_conv_input_channel;

               current_conv_input_height = conv_block[conv_count].size_block[0].input_feature_size[1];
               module_framework.total_framework.max_input_height = (module_framework.total_framework.max_input_height > current_conv_input_height) \
               ? module_framework.total_framework.max_input_height : current_conv_input_height;

               current_conv_input_width = conv_block[conv_count].size_block[0].input_feature_size[2];
               module_framework.total_framework.max_input_width = (module_framework.total_framework.max_input_width > current_conv_input_width) \
               ? module_framework.total_framework.max_input_width : current_conv_input_width;

               current_conv_output_height = conv_block[conv_count].size_block[0].output_feature_size[1];
               module_framework.total_framework.max_output_height = (module_framework.total_framework.max_output_height > current_conv_output_height) \
               ? module_framework.total_framework.max_output_height : current_conv_output_height;

               current_conv_output_width = conv_block[conv_count].size_block[0].output_feature_size[2];
               module_framework.total_framework.max_output_width = (module_framework.total_framework.max_output_width > current_conv_output_width) \
               ? module_framework.total_framework.max_output_width : current_conv_output_width;

               current_filter = conv_block[conv_count].size_block[0].window_size[2];
               module_framework.total_framework.max_filter = (module_framework.total_framework.max_filter > current_filter) \
               ? module_framework.total_framework.max_filter : current_filter;

               if(current_filter == 3) current_window_3_input_channel = current_conv_input_channel;
               if(current_filter == 1) current_window_1_input_channel = current_conv_input_channel;

               module_framework.total_framework.max_filter_size_1 = (module_framework.total_framework.max_filter_size_1  > current_window_1_input_channel) \
               ? module_framework.total_framework.max_filter_size_1 : current_window_1_input_channel;

               module_framework.total_framework.max_filter_size_2 = (module_framework.total_framework.max_filter_size_2  > current_window_3_input_channel) \
               ? module_framework.total_framework.max_filter_size_2 : current_window_3_input_channel;

               module_framework.total_framework.max_bias_size = module_framework.total_framework.max_output_channel;


               //pool operation
               conv_pool_count = conv_count;

               conv_block[conv_pool_count].basic_info_frame.enable_info[2] = true;

               for(int iblob = 0; iblob < pstLayerInfo->iBlobInputNum; iblob++)
               {
                  conv_block[conv_pool_count].size_block[1].input_feature_size[0] = pstLayerInfo->arstInputfeatDim[iblob].C;//C
                  conv_block[conv_pool_count].size_block[1].input_feature_size[1] = pstLayerInfo->arstInputfeatDim[iblob].H;
                  conv_block[conv_pool_count].size_block[1].input_feature_size[2] = pstLayerInfo->arstInputfeatDim[iblob].W;
               }

               conv_block[conv_pool_count].size_block[1].window_size[2] = pstNetPoolparam->ikernel_h;
               conv_block[conv_pool_count].size_block[1].window_size[3] = pstNetPoolparam->ikernel_w;

               for(int iblob = 0; iblob < pstLayerInfo->iBlobOutputNum; iblob++)
               {
                  conv_block[conv_pool_count].size_block[1].output_feature_size[0] = pstLayerInfo->arstOutputfeatDim[iblob].C;//N

                  if(conv_block[conv_pool_count].size_block[1].window_size[2] == conv_block[conv_pool_count].size_block[0].output_feature_size[1])//for end pool
                  {
                     conv_block[conv_pool_count].size_block[1].output_feature_size[1] = conv_block[conv_pool_count].size_block[0].output_feature_size[1];//H+
                     conv_block[conv_pool_count].size_block[1].output_feature_size[2] = conv_block[conv_pool_count].size_block[0].output_feature_size[2];//W+
                  }
                  else
                  {
                     conv_block[conv_pool_count].size_block[1].output_feature_size[1] = pstLayerInfo->arstOutputfeatDim[iblob].H;//H+
                     conv_block[conv_pool_count].size_block[1].output_feature_size[2] = pstLayerInfo->arstOutputfeatDim[iblob].W;//W+
                  }
               
               }

               conv_block[conv_pool_count].size_block[1].window_size[2] = (conv_block[conv_pool_count].size_block[1].window_size[2]==7) ? 3 : conv_block[conv_pool_count].size_block[1].window_size[2]; //modify 7 =>3 for pool window size

               conv_block[conv_pool_count].size_block[1].window_size[0] = conv_block[conv_pool_count].size_block[1].output_feature_size[0];//N
               conv_block[conv_pool_count].size_block[1].window_size[1] = conv_block[conv_pool_count].size_block[1].input_feature_size[0];//C

               conv_block[conv_pool_count].size_block[1].stride_size = pstNetPoolparam->istride_h;
               conv_block[conv_pool_count].size_block[1].pad_size = pstNetPoolparam->ipad_left;

               //get max value
               current_pool_input_height = conv_block[conv_pool_count].size_block[1].input_feature_size[1];
               module_framework.total_framework.max_pool_input_height = (module_framework.total_framework.max_pool_input_height > current_pool_input_height) \
               ? module_framework.total_framework.max_pool_input_height : current_pool_input_height;

               current_pool_input_width = conv_block[conv_pool_count].size_block[1].input_feature_size[2];
               module_framework.total_framework.max_pool_input_width = (module_framework.total_framework.max_pool_input_width > current_pool_input_width) \
               ? module_framework.total_framework.max_pool_input_width : current_pool_input_width;

               current_pool_output_height = conv_block[conv_pool_count].size_block[1].output_feature_size[1];
               module_framework.total_framework.max_pool_output_height = (module_framework.total_framework.max_pool_output_height > current_pool_output_height) \
               ? module_framework.total_framework.max_pool_output_height : current_pool_output_height;

               current_pool_output_width = conv_block[conv_pool_count].size_block[1].output_feature_size[2];
               module_framework.total_framework.max_pool_output_width = (module_framework.total_framework.max_pool_output_width > current_pool_output_width) \
               ? module_framework.total_framework.max_pool_output_width : current_pool_output_width;

               current_pool_window_size = conv_block[conv_pool_count].size_block[1].window_size[2];
               module_framework.total_framework.max_pool_window_size = (module_framework.total_framework.max_pool_window_size > current_pool_window_size) \
               ? module_framework.total_framework.max_pool_window_size : current_pool_window_size;

               current_pool_pad_size = conv_block[conv_pool_count].size_block[1].pad_size;
               module_framework.total_framework.max_pool_pad_size = (module_framework.total_framework.max_pool_pad_size > current_pool_pad_size) \
               ? module_framework.total_framework.max_pool_pad_size : current_pool_pad_size;

            }
            else
            {
               int j_start;
               int conv_siblings_num;
               if((pstLayerInfo->iBlobInputNum ==  1) && !is_concate[input_id])
               {
                  j_start = -1;
                  conv_siblings_num = 1;
               }
               else //for concatenate layer
               {
                  j_start = 0;
                  conv_siblings_num = conv_siblings[conv_count].size();
               }

               for(int j=0; j < conv_siblings_num; j++) 
               {
                  if(is_concate[input_id]) is_inherit_concate[i] = true; //for pool after concate situation

                  conv_pool_count = (j_start==-1) ? conv_count : conv_siblings[conv_count][j];

                  conv_block[conv_pool_count].basic_info_frame.enable_info[2] = (pstNetPoolparam->enPoolMethod == PoolAvg) ? 0 : 1;

                  for(int iblob = 0; iblob < pstLayerInfo->iBlobInputNum; iblob++)
                  {
                     conv_block[conv_pool_count].size_block[1].input_feature_size[0] = pstLayerInfo->arstInputfeatDim[iblob].C;//C
                     conv_block[conv_pool_count].size_block[1].input_feature_size[1] = pstLayerInfo->arstInputfeatDim[iblob].H;
                     conv_block[conv_pool_count].size_block[1].input_feature_size[2] = pstLayerInfo->arstInputfeatDim[iblob].W;
                  }

                  conv_block[conv_pool_count].size_block[1].window_size[2] = pstNetPoolparam->ikernel_h;
                  conv_block[conv_pool_count].size_block[1].window_size[3] = pstNetPoolparam->ikernel_w;

                  for(int iblob = 0; iblob < pstLayerInfo->iBlobOutputNum; iblob++)
                  {
                     conv_block[conv_pool_count].size_block[1].output_feature_size[0] = pstLayerInfo->arstOutputfeatDim[iblob].C;//N
                     if(conv_block[conv_pool_count].size_block[1].window_size[2] == conv_block[conv_pool_count].size_block[0].output_feature_size[1])//for end pool
                     {
                        conv_block[conv_pool_count].size_block[1].output_feature_size[1] = conv_block[conv_pool_count].size_block[0].output_feature_size[1];//H+
                        conv_block[conv_pool_count].size_block[1].output_feature_size[2] = conv_block[conv_pool_count].size_block[0].output_feature_size[2];//W+
                     }
                     else
                     {
                        conv_block[conv_pool_count].size_block[1].output_feature_size[1] = pstLayerInfo->arstOutputfeatDim[iblob].H;//H+
                        conv_block[conv_pool_count].size_block[1].output_feature_size[2] = pstLayerInfo->arstOutputfeatDim[iblob].W;//W+
                     }
                  }

                  conv_block[conv_pool_count].size_block[1].window_size[2] = (conv_block[conv_pool_count].size_block[1].window_size[2]==7) ? 3 : conv_block[conv_pool_count].size_block[1].window_size[2]; 
                  conv_block[conv_pool_count].size_block[1].window_size[3] = (conv_block[conv_pool_count].size_block[1].window_size[3]==7) ? 3 : conv_block[conv_pool_count].size_block[1].window_size[3]; 

                  conv_block[conv_pool_count].size_block[1].window_size[0] = conv_block[conv_pool_count].size_block[1].output_feature_size[0];//N
                  conv_block[conv_pool_count].size_block[1].window_size[1] = conv_block[conv_pool_count].size_block[1].input_feature_size[0];//C

                  conv_block[conv_pool_count].size_block[1].stride_size = pstNetPoolparam->istride_h;
                  conv_block[conv_pool_count].size_block[1].pad_size = pstNetPoolparam->ipad_left;

                  //get max value
                  current_pool_output_channel = conv_block[conv_pool_count].size_block[1].output_feature_size[0]; //take pool into account for max_output_channel
                  module_framework.total_framework.max_output_channel = (module_framework.total_framework.max_output_channel > current_pool_output_channel) \
                  ? module_framework.total_framework.max_output_channel : current_pool_output_channel;

                  current_pool_input_channel = conv_block[conv_pool_count].size_block[1].input_feature_size[0]; //take pool into account for max_input_channel
                  module_framework.total_framework.max_input_channel = (module_framework.total_framework.max_input_channel > current_pool_input_channel) \
                  ? module_framework.total_framework.max_input_channel : current_pool_input_channel;

                  current_pool_input_height = conv_block[conv_pool_count].size_block[1].input_feature_size[1];
                  module_framework.total_framework.max_pool_input_height = (module_framework.total_framework.max_pool_input_height > current_pool_input_height) \
                  ? module_framework.total_framework.max_pool_input_height : current_pool_input_height;

                  current_pool_input_width = conv_block[conv_pool_count].size_block[1].input_feature_size[2];
                  module_framework.total_framework.max_pool_input_width = (module_framework.total_framework.max_pool_input_width > current_pool_input_width) \
                  ? module_framework.total_framework.max_pool_input_width : current_pool_input_width;

                  current_pool_output_height = conv_block[conv_pool_count].size_block[1].output_feature_size[1];
                  module_framework.total_framework.max_pool_output_height = (module_framework.total_framework.max_pool_output_height > current_pool_output_height) \
                  ? module_framework.total_framework.max_pool_output_height : current_pool_output_height;

                  current_pool_output_width = conv_block[conv_pool_count].size_block[1].output_feature_size[2];
                  module_framework.total_framework.max_pool_output_width = (module_framework.total_framework.max_pool_output_width > current_pool_output_width) \
                  ? module_framework.total_framework.max_pool_output_width : current_pool_output_width;

                  current_pool_window_size = conv_block[conv_pool_count].size_block[1].window_size[2];
                  module_framework.total_framework.max_pool_window_size = (module_framework.total_framework.max_pool_window_size > current_pool_window_size) \
                  ? module_framework.total_framework.max_pool_window_size : current_pool_window_size;

                  current_pool_pad_size = conv_block[conv_pool_count].size_block[1].pad_size;
                  module_framework.total_framework.max_pool_pad_size = (module_framework.total_framework.max_pool_pad_size > current_pool_pad_size) \
                  ? module_framework.total_framework.max_pool_pad_size : current_pool_pad_size;

                  conv_block[conv_pool_count].basic_info_frame.specific_struct_enable[0] = (pstNetPoolparam->enPoolMethod == PoolAvg) ? 1 : 0; //kEndPoolEnable
                  module_framework.total_framework.max_bias_size = module_framework.total_framework.max_output_channel; //take pool into account for max_input_channel
                  if(j!= conv_siblings_num - 1)
                  {
                     if(pstNetPoolparam->enPoolMethod == PoolAvg) //poolavg and not the last conv_count
                     {
                        module_framework.conv_blocks[conv_pool_count].basic_info_frame.specific_struct_enable[0] = conv_block[conv_pool_count].basic_info_frame.specific_struct_enable[0];
                     }
                     else
                     {
                        module_framework.conv_blocks[conv_pool_count].basic_info_frame.enable_info[2] = conv_block[conv_pool_count].basic_info_frame.enable_info[2];
                        module_framework.conv_blocks[conv_pool_count].size_block[1].output_feature_size[1] = conv_block[conv_pool_count].size_block[1].output_feature_size[1];
                        module_framework.conv_blocks[conv_pool_count].size_block[1].output_feature_size[2] = conv_block[conv_pool_count].size_block[1].output_feature_size[2];
                     }
                     
                  }
               }
            }
         }
         if(pstOpInfo->enumOpType==OpEltwise)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StEltwiseParam);
				StEltwiseParam *pstNetEltwiseparam;
				pstNetEltwiseparam=(StEltwiseParam*)(pstOpInfo->pvOpInfo);
				
			}
			if(pstOpInfo->enumOpType==OpFc)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StFcParam);
				StFcParam *pstNetFcparam;
				pstNetFcparam=(StFcParam*)(pstOpInfo->pvOpInfo);

            conv_count++;

            id_conv[i] = conv_count;

            if(input_id != -1) 
            {
               if(block_siblings[input_id].id.size() == 1)
               {
                  if(pstLayerInfo->iBlobInputNum > 1) cache_num_max = common_buffer_num;
                  block_siblings[input_id].cache_read_base_num = last_cache_num = (last_cache_num < cache_num_max) ? ++last_cache_num : reset_cache_num;
                  
               }
               else if (block_siblings[input_id].id.size() > 1)
               {
                  cache_num_max = max_cache_read_base;
                  reset_input_cache_num = block_siblings[input_id].cache_read_base_num;
                  last_cache_num = 2;//reserve the buffer 2
               }
            }
            else
            {
               /* code */
               ++last_cache_num;
            }
            
            // conv_process_flag = true;
            if(last_input_layer_id != pstLayerInfo->ariInputBlobId[0])
            {
               layer_num ++;
               last_input_layer_id = pstLayerInfo->ariInputBlobId[0];
            }
            if(conv_count >= 1) module_framework.conv_blocks.push_back(conv_block[conv_count - 1]);//push back last block
            
            conv_block[conv_count].basic_info_frame.specific_struct_enable[5] = pstNetFcparam->bias_term; //for bias enable info
            for(int iblob = 0; iblob < pstLayerInfo->iBlobInputNum; iblob++)
            {
               conv_block[conv_count].size_block[0].input_feature_size[0] = pstLayerInfo->arstInputfeatDim[iblob].C;//C
               conv_block[conv_count].size_block[0].input_feature_size[1] = pstLayerInfo->arstInputfeatDim[iblob].H;
               conv_block[conv_count].size_block[0].input_feature_size[2] = pstLayerInfo->arstInputfeatDim[iblob].W;
            }
            for(int iblob = 0; iblob < pstLayerInfo->iBlobOutputNum; iblob++)
            {
               conv_block[conv_count].size_block[0].output_feature_size[0] = pstLayerInfo->arstOutputfeatDim[iblob].C;//N
               conv_block[conv_count].size_block[0].output_feature_size[1] = pstLayerInfo->arstOutputfeatDim[iblob].H;//H+
               conv_block[conv_count].size_block[0].output_feature_size[2] = pstLayerInfo->arstOutputfeatDim[iblob].W;//W+


               conv_block[conv_count].size_block[1].input_feature_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//C
               conv_block[conv_count].size_block[1].input_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];
               conv_block[conv_count].size_block[1].input_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];

               conv_block[conv_count].size_block[1].output_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];;//H+ for pool initialization
               conv_block[conv_count].size_block[1].output_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];//W+
            }

            conv_block[conv_count].size_block[0].window_size[0] = conv_block[conv_count].size_block[0].output_feature_size[0];//N
            conv_block[conv_count].size_block[0].window_size[1] = conv_block[conv_count].size_block[0].input_feature_size[0];//C
            conv_block[conv_count].size_block[0].window_size[2] = 1;
            conv_block[conv_count].size_block[0].window_size[3] = 1;

            if((strcmp(netname.c_str(),"resnet50") || strcmp(netname.c_str(),"googlenet")))
            {
               conv_block[conv_count].size_block[0].stride_size = 1;
            }
            else
            {
               /* code */
               conv_block[conv_count].size_block[0].stride_size = 0;
            }
            
            conv_block[conv_count].size_block[0].pad_size = 0;

            conv_block[conv_count].basic_info_frame.base_num[0] = (input_id == -1) ? 1 : block_siblings[input_id].cache_read_base_num; //add cachereadbase value
            conv_block[conv_count].basic_info_frame.concate_offset[1] = conv_block[conv_count].size_block[0].output_feature_size[0]; //initialize the kNEnd value
            
            if(input_id != -1) 
            {
               if(block_siblings[input_id].id.size() == 1)
               {
                  if((!strcmp(netname.c_str(),"googlenet")))
                  {
                     if(!multi_write_cache_flag) last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                     else last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - conv_block[conv_count - 1].basic_info_frame.base_num[1];
                  }
                  else
                  {
                     /* code */
                     last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                  }

                  conv_block[conv_count].index_input_layer = conv_count;
                  if(is_concate[input_id] || is_inherit_concate[input_id]) conv_block[conv_count].input_concate_connection = true;
                  module_framework.conv_blocks[input_id].input_concate_connection = conv_block[input_id].input_concate_connection; //needed or not?
               }
               else if(block_siblings[input_id].id.size() > 1)
               {
                  if((!strcmp(netname.c_str(),"googlenet")))
                  conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + last_sub_cache_num - last_sub_cache_num;
                  else
                  {
                     /* code */
                     last_sub_cache_num = conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
                  }
                  

                  int id_first_sibling = block_siblings[input_id].id[0];
                  int conv_first_sibling = id_conv[id_first_sibling];
                  if(!is_concate[input_id] && !is_inherit_concate[input_id])
                  conv_block[conv_count].index_input_layer = id_conv[id_first_sibling];
                  else
                  {
                     conv_block[conv_first_sibling].input_concate_connection = conv_block[conv_count].input_concate_connection = true;
                     module_framework.conv_blocks[conv_first_sibling].input_concate_connection = conv_block[conv_first_sibling].input_concate_connection;
                  }
               }
            }
            else
            {
               /* code */
               conv_block[conv_count].basic_info_frame.base_num[1] = cache_num_max + 1 - conv_block[conv_count].basic_info_frame.base_num[0];
               conv_block[conv_count].index_input_layer = conv_count;
            }
            
            conv_block[conv_count].basic_info_frame.mem_write_enable[0] = 1; //cache write enable
            //get max value
            current_conv_output_channel = conv_block[conv_count].size_block[0].output_feature_size[0];
            module_framework.total_framework.max_output_channel = (module_framework.total_framework.max_output_channel > current_conv_output_channel) \
            ? module_framework.total_framework.max_output_channel : current_conv_output_channel;

            current_conv_input_channel = conv_block[conv_count].size_block[0].input_feature_size[0];
            module_framework.total_framework.max_input_channel = (module_framework.total_framework.max_input_channel > current_conv_input_channel) \
            ? module_framework.total_framework.max_input_channel : current_conv_input_channel;

            current_conv_input_height = conv_block[conv_count].size_block[0].input_feature_size[1];
            module_framework.total_framework.max_input_height = (module_framework.total_framework.max_input_height > current_conv_input_height) \
            ? module_framework.total_framework.max_input_height : current_conv_input_height;

            current_conv_input_width = conv_block[conv_count].size_block[0].input_feature_size[2];
            module_framework.total_framework.max_input_width = (module_framework.total_framework.max_input_width > current_conv_input_width) \
            ? module_framework.total_framework.max_input_width : current_conv_input_width;

            current_conv_output_height = conv_block[conv_count].size_block[0].output_feature_size[1];
            module_framework.total_framework.max_output_height = (module_framework.total_framework.max_output_height > current_conv_output_height) \
            ? module_framework.total_framework.max_output_height : current_conv_output_height;

            current_conv_output_width = conv_block[conv_count].size_block[0].output_feature_size[2];
            module_framework.total_framework.max_output_width = (module_framework.total_framework.max_output_width > current_conv_output_width) \
            ? module_framework.total_framework.max_output_width : current_conv_output_width;

            current_filter = conv_block[conv_count].size_block[0].window_size[2];
            module_framework.total_framework.max_filter = (module_framework.total_framework.max_filter > current_filter) \
            ? module_framework.total_framework.max_filter : current_filter;

            if(current_filter == 3) current_window_3_input_channel = current_conv_input_channel;
            if(current_filter == 1) current_window_1_input_channel = current_conv_input_channel;

            module_framework.total_framework.max_filter_size_1 = (module_framework.total_framework.max_filter_size_1  > current_window_1_input_channel) \
            ? module_framework.total_framework.max_filter_size_1 : current_window_1_input_channel;

            module_framework.total_framework.max_filter_size_2 = (module_framework.total_framework.max_filter_size_2  > current_window_3_input_channel) \
            ? module_framework.total_framework.max_filter_size_2 : current_window_3_input_channel;

            module_framework.total_framework.max_bias_size = module_framework.total_framework.max_output_channel;

			}
         if(pstOpInfo->enumOpType==OpSoftmax)
			{

			}	
                        
         if(pstOpInfo->enumOpType==OpConcat)
         {
            is_concate[i] = true;
         }
         
         if(pstOpInfo->enumOpType==OpRelu)
         {
            if(id_conv[i]!=-1) conv_block[conv_count].basic_info_frame.enable_info[1] = true;
         }

         if(pstOpInfo->enumOpType==OpUnsupport)
         {
            is_concate[i] = is_concate[input_id];  //shortcut operation for unsupport operation
            is_inherit_concate[i] = is_inherit_concate[input_id];
         }

         if((!strcmp(netname.c_str(),"resnet50"))||(!strcmp(netname.c_str(),"googlenet"))) //check every operation
         {
            if((pstLayerInfo->pstFpgaOpInfo->enumOpType == OpConv) || (pstLayerInfo->pstFpgaOpInfo->enumOpType == OpPool) || (pstLayerInfo->pstFpgaOpInfo->enumOpType == OpFc))
            {
               if((conv_block[conv_count].size_block[0].stride_size == 2) && (conv_block[conv_count].size_block[1].stride_size != 2)) //conv_stride is 2 and pool stride is not 2
               {
                  conv_block[conv_count].size_block[0].output_feature_size[1] = conv_block[conv_count - 1].size_block[0].output_feature_size[1]; //tempararily for height
                  conv_block[conv_count].size_block[0].output_feature_size[2] = conv_block[conv_count - 1].size_block[0].output_feature_size[2]; //move following pool stride
                  module_framework.conv_blocks[conv_count].size_block[0].output_feature_size[1] = conv_block[conv_count].size_block[0].output_feature_size[1];
                  module_framework.conv_blocks[conv_count].size_block[0].output_feature_size[2] = conv_block[conv_count].size_block[0].output_feature_size[2];

                  module_framework.conv_blocks[conv_count].size_block[1].input_feature_size[2] = conv_block[conv_count].size_block[1].input_feature_size[2];
                  module_framework.conv_blocks[conv_count].size_block[1].output_feature_size[2] = conv_block[conv_count].size_block[1].output_feature_size[2];
               }

               if(conv_block[conv_count].size_block[1].window_size[2] == 0) conv_block[conv_count].size_block[1].window_size[2] = conv_block[conv_count - 1].size_block[1].window_size[2]; //for pool window postprocessing
            }
         }
      }

      for(int iblob_siblings = 0; iblob_siblings < pstLayerInfo->iBlobInputNum; iblob_siblings++)
      {
         int last_sibling_id = pstLayerInfo->iBlobInputNum -1;
         
         int id_check = (pstLayerInfo->ariInputBlobId[iblob_siblings] == -1) ? 0 : pstLayerInfo->ariInputBlobId[iblob_siblings];
         int id_last = (pstLayerInfo->ariInputBlobId[last_sibling_id] == -1) ? 0 : pstLayerInfo->ariInputBlobId[last_sibling_id];
         if(id_conv[id_check] != -1)
         {
            int conv_check = id_conv[id_check];
            int last_conv = id_conv[id_last];
            conv_siblings[last_conv].push_back(conv_check);
         }

      }

      if(pstLayerInfo->iBlobInputNum > 1)
      {
         if(pstLayerInfo->pstFpgaOpInfo->enumOpType == OpConcat)
         index_concate ++;

         for(int j=0; j<conv_siblings[conv_count].size(); j++) 
         {
            if(conv_count>1) 
            {
               int conv_ddr_count = conv_siblings[conv_count][j];

               if(conv_ddr_count == 1)
               {
                  common_read_ddr_num = 1;
                  common_write_ddr_num = 0;
               }
               else if(conv_ddr_count == 4)
               {
                  common_read_ddr_num = 0;
                  common_write_ddr_num = 2;
               }
               else
               {
                  /* code */
                  common_write_ddr_num = common_read_ddr_num = 2;
               }
               
               last_ddr_read_num = conv_block[conv_ddr_count].basic_info_frame.base_num[2] = (!strcmp(netname.c_str(),"googlenet")) ? 0 : (common_read_ddr_num + 1 - last_ddr_read_num);
               conv_block[conv_ddr_count].basic_info_frame.base_num[3] = (!strcmp(netname.c_str(),"googlenet")) ? 0 : (common_write_ddr_num + 1 - conv_block[conv_ddr_count].basic_info_frame.base_num[2]);
               conv_block[conv_ddr_count].basic_info_frame.mem_write_enable[1] = (!strcmp(netname.c_str(),"googlenet")) ? 0 : 1;
               module_framework.conv_blocks[conv_ddr_count].basic_info_frame.mem_write_enable[1] = conv_block[conv_ddr_count].basic_info_frame.mem_write_enable[1]; //cache write enable for branchs
               
               if(j != (conv_siblings[conv_count].size() - 1)) //modify as no pushing in advance
               {
                  module_framework.conv_blocks[conv_ddr_count].basic_info_frame.base_num[2] = conv_block[conv_ddr_count].basic_info_frame.base_num[2];
                  module_framework.conv_blocks[conv_ddr_count].basic_info_frame.base_num[3] = conv_block[conv_ddr_count].basic_info_frame.base_num[3];

                  module_framework.conv_blocks[conv_ddr_count].basic_info_frame.mem_write_enable[0] = (!strcmp(netname.c_str(),"googlenet")) ? 1 : 0; //cache write disable for branch 1
               }
               else
               {
                  /* code */
                  conv_block[conv_ddr_count].basic_info_frame.specific_struct_enable[1] = (!strcmp(netname.c_str(),"googlenet")) ? 0 : 1; //for addtion layer , tempararily without flag indition
                  conv_block[conv_ddr_count].basic_info_frame.specific_struct_enable[3] = (!strcmp(netname.c_str(),"googlenet")) ? 0 : 1; //for addtion relu layer, tempararily without flag indition
               }

               if(pstLayerInfo->pstFpgaOpInfo->enumOpType == OpConcat) //check concatenation
               {
                  conv_block[conv_ddr_count].basic_info_frame.specific_struct_enable[2] = 1;
                  module_framework.conv_blocks[conv_ddr_count].basic_info_frame.specific_struct_enable[2] = conv_block[conv_ddr_count].basic_info_frame.specific_struct_enable[2];

                  if(j>0)
                  {
                     int conv_offset_start_count = conv_siblings[conv_count][j-1];
                     conv_block[conv_ddr_count].basic_info_frame.concate_offset[0] = conv_block[conv_offset_start_count].size_block[0].output_feature_size[0] + conv_block[conv_offset_start_count].basic_info_frame.concate_offset[0]; //for kNStart
                     module_framework.conv_blocks[conv_ddr_count].basic_info_frame.concate_offset[0] = conv_block[conv_ddr_count].basic_info_frame.concate_offset[0];

                     conv_block[conv_ddr_count].basic_info_frame.concate_offset[1] = conv_block[conv_ddr_count].size_block[0].output_feature_size[0] + conv_block[conv_offset_start_count].basic_info_frame.concate_offset[1]; //for kNEnd
                     module_framework.conv_blocks[conv_ddr_count].basic_info_frame.concate_offset[1] = conv_block[conv_ddr_count].basic_info_frame.concate_offset[1];
                  }
                  else if(j==0)
                  {
                     /* code */
                     conv_block[conv_ddr_count].basic_info_frame.concate_offset[1] = conv_block[conv_ddr_count].size_block[0].output_feature_size[0];
                     module_framework.conv_blocks[conv_ddr_count].basic_info_frame.concate_offset[1] = conv_block[conv_ddr_count].basic_info_frame.concate_offset[1];
                  }
                  
               }

               conv_block[conv_ddr_count].concate_index = (index_concate!=-1) ? index_concate : 0;
               module_framework.conv_blocks[conv_ddr_count].concate_index = conv_block[conv_ddr_count].concate_index;

            }
         }

      }

      module_framework.total_framework.num_concate = index_concate + 1; //as start from -1 from index_concate

      // if((!strcmp(netname.c_str(),"resnet50")) && (conv_count == 0 || conv_count == 10 || conv_count == 23 || conv_count == 42)) //committed out according to latest hardware code
      //{
      //   conv_block[conv_count].wait_cycle = 1000;
      //}

      if(conv_count >= 1 && i == (num_block-1)) module_framework.conv_blocks.push_back(conv_block[conv_count]);//push back the last block
   }

   module_framework.total_framework.num_layer = layer_num;
    
   module_framework.total_framework.num_cnct = conv_count + 1;   
   module_framework.total_framework.num_conv = module_framework.total_framework.num_cnct;                         

    for(int i=0; i<num_block; i++)
    {
       if(is_concate[i]) input_index_base ++;

       if(id_conv[i] != -1)
       {
          int conv_index = id_conv[i];
          if(module_framework.conv_blocks[conv_index].input_concate_connection)
          {  
             conv_block[conv_index].index_input_layer = module_framework.total_framework.num_conv + input_index_base;
             module_framework.conv_blocks[conv_index].index_input_layer =  conv_block[conv_index].index_input_layer;
          }

          if((module_framework.conv_blocks[conv_index].concate_index != 0) && (module_framework.conv_blocks[conv_index].size_block[1].stride_size == 2)) //for kPoolStride2
          {
               for(int j=0; j<conv_siblings[conv_index].size(); j++)
               {
                  int conv_pool_stride2_index = conv_siblings[conv_index][j];
                  conv_block[conv_pool_stride2_index].size_block[1].stride_size = 2;
                  module_framework.conv_blocks[conv_pool_stride2_index].size_block[1].stride_size = conv_block[conv_pool_stride2_index].size_block[1].stride_size;
               }
          }
       }
    }

    module_framework = cycles_computation(module_framework);

    return module_framework;
}

bool ParamGeneration(module_framework_data module_framework, std::string netname) {
     FILE *fp;
   //   fp = fopen("./inc/cnn_param.h","w+");
   //   std::string file_addr = "./inc/"+netname+".h";
     std::string file_addr = "../generated_config_file/"+netname+".h";
   //   fp = fopen("./inc/"+netname.c_str()+".h","w+");
     fp = fopen(file_addr.c_str(),"w+");
     if(fp == NULL)
     {
         printf("Error: fail to create tf2 parameter file.\n");
         return false;
     } 

     transform(netname.begin(), netname.end(), netname.begin(), ::toupper);
     fprintf(fp, "/* Copyright 2019 Inspur Corporation. All Rights Reserved.\n");
     fprintf(fp, "\n");
     fprintf(fp, "Licensed under the Apache License, Version 2.0 (the \"License\");\n");
     fprintf(fp, "you may not use this file except in compliance with the License.\n");
     fprintf(fp, "You may obtain a copy of the License at\n");
     fprintf(fp, "\n");
     fprintf(fp, "    http://www.apache.org/licenses/LICENSE-2.0\n");
     fprintf(fp, "\n");
     fprintf(fp, "Unless required by applicable law or agreed to in writing, software\n");
     fprintf(fp, "distributed under the License is distributed on an \"AS IS\" BASIS,\n");
     fprintf(fp, "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n");
     fprintf(fp, "See the License for the specific language governing permissions and\n");
     fprintf(fp, "limitations under the License.\n");
     fprintf(fp, "==============================================================================*/\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "#ifndef __%s__\n", netname.c_str());
     fprintf(fp, "#define __%s__\n", netname.c_str());
     fprintf(fp, "\n");
     fprintf(fp, "//--------------------------------------------------------------------//\n");
     fprintf(fp, "//                                                                    //\n");
     fprintf(fp, "//             KERNEL/HOST COMMON CNN PARAMETERS                      //\n");
     fprintf(fp, "//                                                                    //\n");
     fprintf(fp, "//                                                                    //\n");
     fprintf(fp, "//--------------------------------------------------------------------//\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "//\n");
     fprintf(fp, "// Debug Parameters\n");
     fprintf(fp, "//\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "//#define CONCAT_LAYER_DEBUG\n");
     fprintf(fp, "\n");
     fprintf(fp, "//#define STATIC_CYCLE\n");
     fprintf(fp, "//#define PRINT_N 1\n");
     fprintf(fp, "//#define PRINT_CYCLE\n");
     fprintf(fp, "#define PRINT_SEQUENCER_INDEX\n");
     fprintf(fp, "//#define STATIC_CYCLE\n");
     fprintf(fp, "//#define PRINT_N 1\n");
     fprintf(fp, "//#define PRINT_CYCLE\n");
     fprintf(fp, "//#define PRINT_N 1\n");
     fprintf(fp, "//#define PRINT_CYCLE\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "//\n");
     fprintf(fp, "// Debug Parameters\n");
     fprintf(fp, "//\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "//============================total summary===========================//\n");
     fprintf(fp, "// number of convolution layers\n");
     fprintf(fp, "#define NUM_LAYER (%d) //for debug\n", module_framework.total_framework.num_conv);
     fprintf(fp, "\n");
     fprintf(fp, "#define NUM_CONVOLUTIONS (%d)\n", module_framework.total_framework.num_conv);
     fprintf(fp, "\n");
     fprintf(fp, "#define NUM_Q_LAYERS (NUM_CONVOLUTIONS + %d + 1)\n", module_framework.total_framework.num_concate);
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define INPUT_IMAGE_C %d\n", module_framework.total_framework.input_image_channel);
     fprintf(fp, "\n");
     fprintf(fp, "#define INPUT_IMAGE_H %d\n", module_framework.total_framework.input_image_height);
     fprintf(fp, "\n");
     fprintf(fp, "#define INPUT_IMAGE_W %d\n", module_framework.total_framework.input_image_width);
     fprintf(fp, "\n");
     fprintf(fp, "#define FIRST_FILTER_SIZE %d\n", module_framework.total_framework.first_filter_size);
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define MAX_OUT_CHANNEL %d\n", module_framework.total_framework.max_output_channel);
     fprintf(fp, "\n");
     fprintf(fp, "#define MAX_POOL_OUTPUT_WVEC CEIL(%d, W_VECTOR)\n",module_framework.total_framework.max_pool_output_width);
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "// the maximum pool window size \n");
     fprintf(fp, "\n");
     fprintf(fp, "#define POOL_WINDOW_MAX %d\n", module_framework.total_framework.max_pool_window_size);
     fprintf(fp, "\n");
     fprintf(fp, "// set size of input_size \n");
     int current_max_ddr_size = 0;
     int ddr_window_size = 0;
     int ddr_pool_output_width = 0;
     int ddr_pool_output_height = 0;
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++){
        if(module_framework.conv_blocks[c_index].basic_info_frame.mem_write_enable[1] || (!strcmp(netname.c_str(),"GOOGLENET")))
        {
            if(ceil((module_framework.conv_blocks[c_index].size_block[0].window_size[0] * 1.0)/C_VECTOR) * module_framework.conv_blocks[c_index].size_block[1].output_feature_size[1] * ceil((module_framework.conv_blocks[c_index].size_block[1].output_feature_size[2] * 1.0)/W_VECTOR) > current_max_ddr_size)
            {
               ddr_window_size = module_framework.conv_blocks[c_index].size_block[0].window_size[0];
               ddr_pool_output_height = module_framework.conv_blocks[c_index].size_block[1].output_feature_size[1];
               ddr_pool_output_width = module_framework.conv_blocks[c_index].size_block[1].output_feature_size[2];
               current_max_ddr_size = ceil((module_framework.conv_blocks[c_index].size_block[0].window_size[0] * 1.0)/C_VECTOR) * module_framework.conv_blocks[c_index].size_block[1].output_feature_size[1] * ceil((module_framework.conv_blocks[c_index].size_block[1].output_feature_size[2] * 1.0)/W_VECTOR);
            }
        }
     }
     fprintf(fp, "#define DDR_PAGE_SIZE0 (CEIL(%d, C_VECTOR) * %d * CEIL(%d, W_VECTOR))\n",ddr_window_size, ddr_pool_output_width, ddr_pool_output_width);
     fprintf(fp, "#define DDR_PAGE_SIZE1 (CEIL(%d, C_VECTOR) * %d * CEIL(%d, W_VECTOR))\n",ddr_window_size, ddr_pool_output_width, ddr_pool_output_width);
     fprintf(fp, "#define DDR_SIZE (DDR_PAGE_SIZE0 + DDR_PAGE_SIZE1)\n");
     fprintf(fp, "\n");
     int current_max_cache_size = 0;
     int cache_window_size = 0;
     int cache_pool_output_width = 0;
     int cache_pool_output_height = 0;
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++)
     {
        if(module_framework.conv_blocks[c_index].basic_info_frame.mem_write_enable[0])
        {
            if(ceil((module_framework.conv_blocks[c_index].size_block[0].window_size[1] * 1.0)/C_VECTOR) * module_framework.conv_blocks[c_index].size_block[0].input_feature_size[1] * ceil((module_framework.conv_blocks[c_index].size_block[0].input_feature_size[2]*1.0)/W_VECTOR) > current_max_cache_size)
            {
               cache_window_size = module_framework.conv_blocks[c_index].size_block[0].window_size[1];
               cache_pool_output_height = module_framework.conv_blocks[c_index].size_block[0].input_feature_size[1];
               cache_pool_output_width = module_framework.conv_blocks[c_index].size_block[0].input_feature_size[2];
               current_max_cache_size = ceil((module_framework.conv_blocks[c_index].size_block[0].window_size[1] * 1.0) / C_VECTOR) * module_framework.conv_blocks[c_index].size_block[0].input_feature_size[1] * ceil((module_framework.conv_blocks[c_index].size_block[0].input_feature_size[2]*1.0)/W_VECTOR);
            }
        }
     }
     fprintf(fp, "#define CACHE_PAGE_SIZE (CEIL(%d, C_VECTOR) * %d * CEIL(%d, W_VECTOR))\n",cache_window_size, cache_pool_output_height ,cache_pool_output_width);
     if(!strcmp(netname.c_str(),"RESNET50"))
     fprintf(fp, "#define CACHE_SIZE (CACHE_PAGE_SIZE * 2)\n");
     else if(!strcmp(netname.c_str(),"GOOGLENET"))
     fprintf(fp, "#define CACHE_SIZE (CACHE_PAGE_SIZE * 3)\n");
     fprintf(fp, "\n");
     fprintf(fp, "// the largest of conv1 and conv2 filters\n");
     fprintf(fp, "#define FILTER_CACHE_PAGE_SIZE1 (NEXT_DIVISIBLE(%d, C_VECTOR) * 3 * CEIL(3, FW_VECTOR))\n",module_framework.total_framework.max_filter_size_2);
     fprintf(fp, "#define FILTER_CACHE_PAGE_SIZE2 (NEXT_DIVISIBLE(%d, C_VECTOR * FW_VECTOR))\n",module_framework.total_framework.max_filter_size_1);
     fprintf(fp, "#define FILTER_CACHE_PAGE_SIZE  (MYMAX2(FILTER_CACHE_PAGE_SIZE1, FILTER_CACHE_PAGE_SIZE2))\n");
     fprintf(fp, "\n");
     fprintf(fp, "///MODDDS\n");
     fprintf(fp, "#define FILTER_CACHE_PAGE_DEPTH (NEXT_POWER_OF_2(CEIL(FILTER_CACHE_PAGE_SIZE, C_VECTOR)))\n");
     fprintf(fp, "#define FILTER_CACHE_DEPTH (FILTER_CACHE_PAGE_DEPTH * DOUBLE_BUFFER_DIM)\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FILTER_DDR_READ_STEP1 (CEIL((C_VECTOR * FW_VECTOR), DDR_BANDWIDTH_IN_BYTES))\n");
     fprintf(fp, "#define FILTER_DDR_READ_STEP2 (CEIL((C_VECTOR * 1), DDR_BANDWIDTH_IN_BYTES))\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FEATURE_DDR_READ_STEP 1\n");
     fprintf(fp, "\n");
     fprintf(fp, "//Set size of host filter and bias buffer of each layer.\n");
     fprintf(fp,"#define MAX_FILTER_SIZE1 (NEXT_POWER_OF_2(CEIL(%d, C_VECTOR) * 1 * CEIL(1, FW_VECTOR) * NEXT_DIVISIBLE(%d, N_VECTOR) * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))\n", module_framework.total_framework.max_filter_size_1, module_framework.total_framework.max_filter_size_1);
     fprintf(fp,"#define MAX_FILTER_SIZE2 (NEXT_POWER_OF_2(CEIL(%d, C_VECTOR) * 3 * CEIL(3, FW_VECTOR) * NEXT_DIVISIBLE(%d, N_VECTOR) * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))\n", module_framework.total_framework.max_filter_size_2, module_framework.total_framework.max_filter_size_2);
     fprintf(fp,"#define MAX_FILTER_SIZE_TEMP ((MAX_FILTER_SIZE1 > MAX_FILTER_SIZE2) ? MAX_FILTER_SIZE1 : MAX_FILTER_SIZE2)\n");
     fprintf(fp,"#define MAX_FILTER_SIZE (CEIL(MAX_FILTER_SIZE_TEMP, NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR)))\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define MAX_BIAS_SIZE  NEXT_DIVISIBLE(%d, N_VECTOR)\n", module_framework.total_framework.max_bias_size);
     fprintf(fp, "\n");
     fprintf(fp, "// used by pool.cl\n");
     fprintf(fp, "#define EDGE_H (POOL_WINDOW_MAX - 1)\n");
     fprintf(fp, "#define EDGE_W (POOL_WINDOW_MAX - 1)\n");
     fprintf(fp, "#define WVEC_ITER (CEIL(kOwEndWithOffsetMax, OW_VECTOR))\n");
     fprintf(fp, "#define NNVEC_ITER (CEIL(N_VECTOR, NARROW_N_VECTOR))\n");
     fprintf(fp, "#define EDGE_H_BUFFER_SIZE (WVEC_ITER * NNVEC_ITER)\n");
     fprintf(fp, "#define EDGE_W_BUFFER_SIZE (NNVEC_ITER)\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define DDR_BLOCK_SIZE DDR_PAGE_SIZE0\n");
     fprintf(fp, "#define D0 0\n");
     fprintf(fp, "#define D1 0\n");
     fprintf(fp, "#define D2 DDR_BLOCK_SIZE\n");
     fprintf(fp, "\n");
     if(!strcmp(netname.c_str(),"RESNET50"))
     fprintf(fp, "#define OUTPUT_OFFSET (2 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR))\n");
     else if(!strcmp(netname.c_str(),"GOOGLENET"))
     fprintf(fp, "#define OUTPUT_OFFSET (3 * DDR_BLOCK_SIZE * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR))\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define C1 0\n");
     fprintf(fp, "#define C2 CACHE_PAGE_SIZE\n");
     if(!strcmp(netname.c_str(),"GOOGLENET"))
     fprintf(fp, "#define C3 (2 * CACHE_PAGE_SIZE)\n");
     fprintf(fp, "\n");
     fprintf(fp, "//==========================basic structure===========================//\n");
     fprintf(fp, "//----------------------------base addr info-----------------------------//\n");
     fprintf(fp, "CONSTANT int kCacheReadBase[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " C%d", module_framework.conv_blocks[c_index].basic_info_frame.base_num[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kCacheWriteBase[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " C%d", module_framework.conv_blocks[c_index].basic_info_frame.base_num[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kDDRReadBase[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " D%d", module_framework.conv_blocks[c_index].basic_info_frame.base_num[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kDDRWriteBase[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " D%d", module_framework.conv_blocks[c_index].basic_info_frame.base_num[3]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kCacheWriteEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(c_index < module_framework.total_framework.num_conv -1)
        {
			    fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.mem_write_enable[0]);
		    }
		    else
	      {
          fprintf(fp, " %d", 0);//force to set to 0 for last layer
		    }
		    if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kDDRWriteEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.mem_write_enable[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kEndPoolEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kAdditionEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kBranchTail[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kAdditionReluEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[3]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kIpoolEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[4]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT bool kBiasEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[5]) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "//----------------------------enable info-----------------------------//\n");
     //conv_enabled
     fprintf(fp, "CONSTANT bool kConvEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].basic_info_frame.enable_info[0]) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     //relu_enabled
     fprintf(fp, "CONSTANT bool kReluEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].basic_info_frame.enable_info[1]) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     //pool_enabled
     fprintf(fp, "CONSTANT bool kPoolEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].basic_info_frame.enable_info[2]) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     //BN_enabled
     fprintf(fp, "CONSTANT bool kBnEnable[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].basic_info_frame.enable_info[3]) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//----------------------concatenation and conv info-------------------//\n");
     fprintf(fp, "CONSTANT int kConcatLayer[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].concate_index);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kInputLayer[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].index_input_layer);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//-----------------------------wait cycles-------------------------------//\n");
     fprintf(fp, "CONSTANT int kSequencerIdleCycle[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].wait_cycle);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//===========================detail info===============================//\n");
     fprintf(fp, "//-------------------------pool classification-------------------------//\n");
     fprintf(fp, "// 0 - max pooling\n");
     fprintf(fp, "// 1 - average pooling\n");
     fprintf(fp, "CONSTANT int kPoolType[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", (!module_framework.conv_blocks[c_index].pool_max));
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//-------------------------------size info----------------------------//\n");
     fprintf(fp, "//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~conv~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//\n");
     fprintf(fp, "//channel\n");
     fprintf(fp, "CONSTANT int kCvecEnd[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " CEIL(%d, C_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kCvecEndMax = CEIL(%d, C_VECTOR);\n", module_framework.total_framework.max_input_channel);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kFilterCvecEnd[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].size_block[0].window_size[3] == 3)
        {
          fprintf(fp, " CEIL(%d, C_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0]);
        }
        else if(module_framework.conv_blocks[c_index].size_block[0].window_size[3] == 5)
        {
          fprintf(fp, " CEIL(%d, C_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0]);
        }
        else
        {
          fprintf(fp, " CEIL(%d, C_VECTOR * FW_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0]);
        }
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kFilterCvecEndMax = CEIL(%d, C_VECTOR * FW_VECTOR);\n", module_framework.total_framework.max_input_channel);
     fprintf(fp, "\n");
     fprintf(fp, "// input\n");
     fprintf(fp, "CONSTANT int END_WW_MAX_INPUT_READER = CEIL(%d, FW_VECTOR);\n", module_framework.total_framework.max_input_width);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kNvecEnd[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " CEIL(%d, N_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].output_feature_size[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kNvecEndMax = CEIL(%d, N_VECTOR);\n", module_framework.total_framework.max_output_channel);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kNEndWithOffset[%d] = \n", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].output_feature_size[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kNEndWithOffsetMax = %d;\n", module_framework.total_framework.max_output_channel);
     fprintf(fp, "\n");
     fprintf(fp, "//input height\n");
     fprintf(fp, "CONSTANT int kInputHeight[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kInputHeightMax = %d;\n", module_framework.total_framework.max_input_height);
     fprintf(fp, "\n");
     fprintf(fp, "//input width\n");
     fprintf(fp, "CONSTANT int kInputWidth[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kInputWidthMax = %d;\n", module_framework.total_framework.max_input_width);
     fprintf(fp, "\n");
     fprintf(fp, "//output height\n");
     fprintf(fp, "CONSTANT int kOutputHeight[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].output_feature_size[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kOutputHeightMax = %d;\n", module_framework.total_framework.max_output_height);
     fprintf(fp, "\n");
     fprintf(fp, "//output width\n");
     fprintf(fp, "CONSTANT int kOutputWidth[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].output_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kOutputWidthMax = %d;\n", module_framework.total_framework.max_output_width);
     fprintf(fp, "\n");
     fprintf(fp, "//This is the K dimension\n");
     fprintf(fp, "CONSTANT int kOutputChannels[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].window_size[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kOutputChannelsMax = %d;\n", module_framework.total_framework.max_output_channel);
     fprintf(fp, "\n");
     fprintf(fp, "//This is the C dimension\n");
     fprintf(fp, "CONSTANT int kInputChannels[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].window_size[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//window size\n");
     fprintf(fp, "CONSTANT int kFilterSize[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].window_size[3]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kFilterSizeMax = %d;\n", module_framework.total_framework.max_filter);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kFWvecEnd[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " CEIL(%d, FW_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].window_size[3]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kFWvecEndMax = CEIL(%d, FW_VECTOR);\n", module_framework.total_framework.max_filter);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kWvecEnd[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " CEIL(%d, W_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kWvecEndMax = CEIL(%d, W_VECTOR);\n", module_framework.total_framework.max_input_width);
     fprintf(fp, "\n");
     fprintf(fp, "//Conv pad\n");
     fprintf(fp, "CONSTANT int kPoolPad[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[1].pad_size);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kConvStride[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].stride_size);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//Conv pad\n");
     fprintf(fp, "CONSTANT int kPadHeight[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].pad_size);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int  kPadWidth[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[0].pad_size);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "// for pool computation\n");
     fprintf(fp, "CONSTANT int kNStart[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.concate_offset[0]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kNEnd[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].basic_info_frame.concate_offset[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~pool~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//\n");
     fprintf(fp, "//pool output feature map width\n");
     fprintf(fp, "CONSTANT int kOhEndWithOffset[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d + POOL_OFFSET_P", module_framework.conv_blocks[c_index].size_block[1].input_feature_size[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kOhEndWithOffsetMax = %d + POOL_OFFSET_P;\n", module_framework.total_framework.max_pool_input_height);
     fprintf(fp, "\n");
     fprintf(fp, "//pool output feature map hight\n");
     fprintf(fp, "CONSTANT int kOwEndWithOffset[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d + POOL_OFFSET_Q", module_framework.conv_blocks[c_index].size_block[0].output_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kOwEndWithOffsetMax = %d + POOL_OFFSET_Q;\n", module_framework.total_framework.max_output_width);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int  kPoolOutputHeight[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[1].output_feature_size[1]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int  kPoolOutputHeightMax = %d;\n", module_framework.total_framework.max_pool_output_height);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int  kPoolOutputWidth[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[1].output_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "CONSTANT int kPoolOutputWidthMax = %d;\n", module_framework.total_framework.max_pool_output_width);
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kPoolOutputWvecEnd[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " CEIL(%d, W_VECTOR)", module_framework.conv_blocks[c_index].size_block[1].output_feature_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int kPoolWindow[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.conv_blocks[c_index].size_block[1].window_size[2]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT bool kPoolStride2[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].size_block[1].stride_size == 2) fprintf(fp, " true");
        else fprintf(fp, " false");
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//----------------------------others----------------------------------//\n");
     fprintf(fp, "//~~~~~~~~~~~~~~~~~~~~~~~~~~~~offset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//\n");
     fprintf(fp, "// how much filter data (number of float_vec_t reads) we need to prefetch\n");
     fprintf(fp, "// at each stage of convolution\n");
     fprintf(fp, "// formula : num_cvec * R * end_ss\n");
     fprintf(fp, "CONSTANT int kFilterLoadSize[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        if(module_framework.conv_blocks[c_index].size_block[0].window_size[3] == 3)
        {
         fprintf(fp, " CEIL(%d, C_VECTOR) * %d * CEIL(%d, FW_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0], \
         module_framework.conv_blocks[c_index].size_block[0].window_size[3], module_framework.conv_blocks[c_index].size_block[0].window_size[3]);
        }
        else if(module_framework.conv_blocks[c_index].size_block[0].window_size[3] == 5)
        {
         fprintf(fp, " CEIL(%d, C_VECTOR) * %d * CEIL(%d, FW_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0], \
         module_framework.conv_blocks[c_index].size_block[0].window_size[3], module_framework.conv_blocks[c_index].size_block[0].window_size[3]);
        }
        else if(module_framework.conv_blocks[c_index].basic_info_frame.specific_struct_enable[4])
        {
         fprintf(fp, " 0");
        }
        else
        {
         fprintf(fp, " CEIL(%d, C_VECTOR * FW_VECTOR)", module_framework.conv_blocks[c_index].size_block[0].input_feature_size[0], \
         module_framework.conv_blocks[c_index].size_block[0].window_size[3], module_framework.conv_blocks[c_index].size_block[0].window_size[3]);
        } 
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "//===========================cycles computation=======================//\n");
     fprintf(fp, "//\n");
     fprintf(fp, "// static cycles\n");
     fprintf(fp, "//\n");
     fprintf(fp, "#ifdef STATIC_CYCLE\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int feature_writer_cycles[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.cycle_block.feature_writer_cyc[c_index]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int conv_cycles[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.cycle_block.conv_cyc[c_index]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int pool_cycles[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.cycle_block.pool_cyc[c_index]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "CONSTANT int filter_reader_conv_cycles[%d] = ", module_framework.total_framework.num_conv);
     fprintf(fp, "{\n");
     for(int c_index = 0; c_index < module_framework.total_framework.num_conv; c_index++) {
        fprintf(fp, " %d", module_framework.cycle_block.filter_reader_conv_cyc[c_index]);
        if(c_index != module_framework.total_framework.num_conv - 1) fprintf(fp, ",");
     }
     fprintf(fp, "\n");
     fprintf(fp, "};\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FEATURE_WRITER_CYCLE(i) feature_writer_cycles[i]\n");
     fprintf(fp, "#define FILTER_READER_CONV_CYCLE(i) filter_reader_conv_cycles[i]\n");
     fprintf(fp, "#define CONV_CYCLE(i) conv_cycles[i]\n");
     fprintf(fp, "#define POOL_CYCLE(i) pool_cycles[i]\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define CONV_TOTAL_CYCLE %d\n", module_framework.cycle_block.conv_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define INPUT_READER_CYCLE %d\n", module_framework.cycle_block.input_reader_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define FILTER_PRELOAD_CYCLE %d\n", module_framework.cycle_block.filter_preload_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define FILTER_READER_CONV_TOTAL_CYCLE %d\n", module_framework.cycle_block.filter_reader_conv_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define CONV_TOTAL_WRITE_CACHE %d\n", module_framework.cycle_block.write_cache_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define POOL_TOTAL_CYCLE %d\n", module_framework.cycle_block.pool_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define FEATURE_WRITER_TOTAL_CYCLE %d\n", module_framework.cycle_block.feature_writer_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#define END_POOL_TOTAL_CYCLE %d\n", module_framework.cycle_block.end_pool_total_cyc);
     fprintf(fp, "\n");
     fprintf(fp, "#endif\n");
     fprintf(fp, "\n");
     fprintf(fp, "#ifndef STATIC_CYCLE\n");
     fprintf(fp, "\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FEATURE_WRITER_CYCLE(i) FindFeatureWriterCycles(i)\n");
     fprintf(fp, "#define FILTER_READER_CONV_CYCLE(i) FindFilterReaderConvCycles(i)\n");
     fprintf(fp, "#define CONV_CYCLE(i) FindConvCycles(i)\n");
     fprintf(fp, "#define POOL_CYCLE(i) FindPoolCycles(i)\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define CONV_TOTAL_CYCLE FindConvTotalCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define INPUT_READER_CYCLE FindInputReaderCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FILTER_PRELOAD_CYCLE FindFilterPreloadCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FILTER_READER_CONV_TOTAL_CYCLE FindFilterReaderConvTotalCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define CONV_TOTAL_WRITE_CACHE FindConvTotalWriteCache()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define POOL_TOTAL_CYCLE FindPoolTotalCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define FEATURE_WRITER_TOTAL_CYCLE FindFeatureWriterTotalCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#define END_POOL_TOTAL_CYCLE FindEndPoolTotalCycles()\n");
     fprintf(fp, "\n");
     fprintf(fp, "#endif\n");
     fprintf(fp, "\n");
     fprintf(fp, "#endif // __%s__\n", netname.c_str());
     fprintf(fp, "\n");
     fclose(fp);
     return true;
}
