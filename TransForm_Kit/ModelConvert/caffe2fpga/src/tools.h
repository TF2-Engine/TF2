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
#ifndef TOOLS_H
#define TOOLS_H
#include <stdio.h>
#include <limits.h>
#include <vector>
#include <set>
#include <limits>
#include <map>
#include <algorithm>
#include "fpganetworkinterface.h"
typedef struct StBlobInfo
{
	std::string blobName;
	StBlobShape blobshape;
	int         id_index;
}StBlobInfo;
typedef struct StOpInfo
{
	std::string opName;
	EN_fpgaop enfpgaOpType;//fpga type
	void *pvOpParam;
}StOpInfo;
void remove_split(std::map<int,std::vector<StOpInfo>> &layerIndex_opName,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob);
void hebing_op(std::map<int,std::vector<StOpInfo>> &layerIndex_opName,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob);
void compute_op_outputsize(StOpInfo stopInfo,std::vector<StBlobShape> &vc_inputblobshape,std::vector<StBlobShape> &vc_outputblobshape);

int compute_layers_outputsize(StBlobShape stNetInput,std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob);
void map_to_fpganetwork(std::string netName,std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob);
#endif
