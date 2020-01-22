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

#include "fpganetworkinterface.h"
#include <iostream>
#include<string.h>
#define PRINT_EN 0
bool read_fppanetwork(std::string filename,StFpgaNetInfo *pstNetInfo)
{
	int size_memory=0;
	FILE *fp=fopen(filename.c_str(),"rb");
	if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filename.c_str());
        return false;
    }
	//wenjian daxiao
	fseek(fp, 0, SEEK_END);  
    int size = ftell(fp);  
    rewind(fp); 
    //std::cout<<"file size:"<<size<<std::endl;
	//fseek(fp,100L,1);
	void *pvMemory=(void*)pstNetInfo;
	memset(pvMemory,0,size);
	int readreturn=fread(pvMemory,1,size,fp);
	std::cout<<"read return:"<<readreturn<<std::endl;
	fclose(fp);
	//std::cout<<pstNetInfo->archNetName<<"netname:"<<pstNetInfo->iNetworkVersion<<" "<<pstNetInfo->iLayerNum<<std::endl;
	if(PRINT_EN)
	{
		std::cout<<pstNetInfo->archNetName<<"netname:"<<pstNetInfo->iNetworkVersion<<" "<<pstNetInfo->iLayerNum<<std::endl;
	}
	size_memory+=sizeof(StFpgaNetInfo);
	
	//layer info 
	for(int i=0;i<pstNetInfo->iLayerNum;i++)
	{
		
		StFpgaLayerInfo *pstLayerInfo=(StFpgaLayerInfo*)((char*)pvMemory+size_memory);
		size_memory+=sizeof(StFpgaLayerInfo);
		pstNetInfo->pstFpgaLayerInfo=pstLayerInfo;
		if(PRINT_EN)
		{
			std::cout<<"layerindex:"<<pstLayerInfo->iLayerId<<"  "
					<<pstLayerInfo->iOpNum<<"  "
					<<pstLayerInfo->iBlobInputNum<<"  "
					<<pstLayerInfo->iBlobOutputNum<<"  ";	
		}					
		//read input and output size and index
		for(int iblob=0;iblob<pstLayerInfo->iBlobInputNum;iblob++)
		{
			if(PRINT_EN)
			{
				std::cout<<pstLayerInfo->ariInputBlobId[iblob]<<"  "
						 <<pstLayerInfo->arstInputfeatDim[iblob].N<<"  "
						 <<pstLayerInfo->arstInputfeatDim[iblob].C<<"  "
						 <<pstLayerInfo->arstInputfeatDim[iblob].H<<"  "
						 <<pstLayerInfo->arstInputfeatDim[iblob].W<<"  ";
			}
		}
		for(int iblob=0;iblob<pstLayerInfo->iBlobOutputNum;iblob++)
		{
			if(PRINT_EN)
			{
				std::cout<<pstLayerInfo->ariOutputBlobId[iblob]<<"  "
						 <<pstLayerInfo->arstOutputfeatDim[iblob].N<<"  "
						 <<pstLayerInfo->arstOutputfeatDim[iblob].C<<"  "
						 <<pstLayerInfo->arstOutputfeatDim[iblob].H<<"  "
						 <<pstLayerInfo->arstOutputfeatDim[iblob].W<<"  ";
			}
		}
		//layer op info
		for(int iop=0;iop<pstLayerInfo->iOpNum;iop++)
		{
			StFpgaOpInfo *pstOpInfo=(StFpgaOpInfo*)((char*)pvMemory+size_memory);
			pstLayerInfo->pstFpgaOpInfo=pstOpInfo;
			size_memory+=sizeof(StFpgaOpInfo);
			if(PRINT_EN)
			{
				std::cout<<pstOpInfo->enumOpType<<"  ";
			}
			if(pstOpInfo->enumOpType==OpConv)
			{
				//std::cout<<"size memory:"<<size_memory<<std::endl;
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StConvParam);
				StConvParam *pstNetConvparam=(StConvParam*)pstOpInfo->pvOpInfo;
				if(PRINT_EN)
				{
					std::cout<<"channel:"<<pstNetConvparam->iKernelChannel<<",bias_term:"<<pstNetConvparam->bias_term
							 <<",pad:"<<pstNetConvparam->ipad_left<<","<<pstNetConvparam->ipad_top<<","<<pstNetConvparam->ipad_right<<","<<pstNetConvparam->ipad_bottom
							 <<",kernel:"<<pstNetConvparam->ikernel_h<<","<<pstNetConvparam->ikernel_w
							 <<",stride:"<<pstNetConvparam->istride_h<<","<<pstNetConvparam->istride_w<<"  ";
				}
				
			}
			if(pstOpInfo->enumOpType==OpBn)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StBnParam);
				StBnParam *pstNetBnparam;
				pstNetBnparam=(StBnParam*)(pstOpInfo->pvOpInfo);
				if(PRINT_EN)
				{
					std::cout<<pstNetBnparam->fEps<<"  ";
				}
				
			}
			if(pstOpInfo->enumOpType==OpScale)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StScaleParam);
				//StScaleParam *pstNetScaleparam;
				//pstNetScaleparam=(StScaleParam*)(pstOpInfo->pvOpInfo);
				
			}
			if(pstOpInfo->enumOpType==OpPool)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StPoolParam);
				StPoolParam *pstNetPoolparam;
				pstNetPoolparam=(StPoolParam*)(pstOpInfo->pvOpInfo);
				if(PRINT_EN)
				{
					std::cout<<"Pool method:"<<pstNetPoolparam->enPoolMethod
							 <<",pad:"<<pstNetPoolparam->ipad_left<<","<<pstNetPoolparam->ipad_top<<","<<pstNetPoolparam->ipad_right<<","<<pstNetPoolparam->ipad_bottom
							 <<",kernel:"<<pstNetPoolparam->ikernel_h<<","<<pstNetPoolparam->ikernel_w
							 <<",stride:"<<pstNetPoolparam->istride_h<<","<<pstNetPoolparam->istride_w<<"  ";
				}
				
			}
			if(pstOpInfo->enumOpType==OpEltwise)
			{
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StEltwiseParam);
				StEltwiseParam *pstNetEltwiseparam;
				pstNetEltwiseparam=(StEltwiseParam*)(pstOpInfo->pvOpInfo);
				if(PRINT_EN)
				{
					std::cout<<"Eltwise method:"<<pstNetEltwiseparam->enEltwiseMethod<<"  ";
				}
				
			}
			if(pstOpInfo->enumOpType==OpFc)
			{
				
				pstOpInfo->pvOpInfo=(void*)((char*)pvMemory+size_memory);
				size_memory+=sizeof(StFcParam);
				StFcParam *pstNetFcparam;
				pstNetFcparam=(StFcParam*)(pstOpInfo->pvOpInfo);
				if(PRINT_EN)
				{
					std::cout<<"bias term:"<<pstNetFcparam->bias_term<<",outchannel:"<<pstNetFcparam->iKernelChannel<<"  ";
				}
				
			}	
		}//end op
		if(PRINT_EN)
		{
			std::cout<<std::endl;
		}
	}
	return 0;
}
