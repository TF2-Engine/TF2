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

#include "tools.h"
#include <iostream>
//#include <stdlib.h>
#include <math.h>
#include<string.h>
#define UN_CHANGE_OPNUM 3
#define MAX_LAYERNUM 10000
#define DEBUG_MOD 1
std::string unChangeSizeOpName[3]={"BatchNorm","ReLU","Scale"};
void remove_split(std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob)
{
	std::map<int,std::vector<StOpInfo>>::iterator it_layerIndex_opInfo;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_inputBlob;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_outputBlob;
	it_layerIndex_opInfo=layerIndex_opInfo.begin();
	it_layerIndex_inputBlob=layerIndex_inputBlob.begin();
	it_layerIndex_outputBlob=layerIndex_outputBlob.begin();
	while((it_layerIndex_opInfo!=layerIndex_opInfo.end())
		&& (it_layerIndex_inputBlob!=layerIndex_inputBlob.end())
		&& (it_layerIndex_outputBlob!=layerIndex_outputBlob.end()))
	{
		if(it_layerIndex_opInfo->second[0].opName=="Split")
		{
			//std::cout<<it_layerIndex_opInfo->first<<std::endl;
			std::vector<StBlobInfo> inputBlobInfo=it_layerIndex_inputBlob->second;
			std::vector<StBlobInfo> outputBlobInfo=it_layerIndex_outputBlob->second;
			layerIndex_opInfo.erase(it_layerIndex_opInfo++);
			layerIndex_inputBlob.erase(it_layerIndex_inputBlob++);
			layerIndex_outputBlob.erase(it_layerIndex_outputBlob++);
			//change inputblobname
			std::map<int,std::vector<StBlobInfo>>::iterator  itchange_layerIndex_inputBlob;
			itchange_layerIndex_inputBlob=layerIndex_inputBlob.begin();
			int findChangeNum=0;
			while(itchange_layerIndex_inputBlob!=layerIndex_inputBlob.end())
			{
				std::vector<StBlobInfo> changeinputBlobInfo=itchange_layerIndex_inputBlob->second;
				for(int j=0;j<outputBlobInfo.size();j++)
				{
					for(int i=0;i<changeinputBlobInfo.size();i++)
					{
						if(changeinputBlobInfo[i].blobName==outputBlobInfo[j].blobName)
						{
							changeinputBlobInfo[i].blobName=inputBlobInfo[0].blobName;
							//std::cout<<"find:"<<inputBlobInfo[0].blobName<<std::endl;
							itchange_layerIndex_inputBlob->second[i]=changeinputBlobInfo[i];
							findChangeNum++;
							break;
						}
					}
				}
				if(findChangeNum==outputBlobInfo.size())
				{
					break;
				}
				itchange_layerIndex_inputBlob++;
			}
		}
		else
		{
			it_layerIndex_opInfo++;
			it_layerIndex_inputBlob++;
			it_layerIndex_outputBlob++;
		}
	}
}

//jing conv bn scale relu hebing
void hebing_op(std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob)
{
	std::map<int,std::vector<StOpInfo>> hebing_layerIndex_opInfo;//hebing hou layerIndex_opInfo
	int hebing_layerIndex=0;//hebing hou index
	std::map<int,std::vector<StBlobInfo>> hebing_layerIndex_inputBlob;
	std::map<int,std::vector<StBlobInfo>> hebing_layerIndex_outputBlob;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_inputBlob;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_outputBlob;
	std::map<int,std::vector<StOpInfo>>::iterator it_layerIndex_opInfo;
	it_layerIndex_opInfo=layerIndex_opInfo.begin();
	it_layerIndex_inputBlob=layerIndex_inputBlob.begin();
	it_layerIndex_outputBlob=layerIndex_outputBlob.begin();
	std::vector<int> split_index;
	//find the value that not unChangeSizeOpName and record into split_index
	while(it_layerIndex_opInfo!=layerIndex_opInfo.end())
	{
		bool hebing_flag=0;
		for(int i=0;i<UN_CHANGE_OPNUM;i++)
		{
			if(it_layerIndex_opInfo->second[0].opName==unChangeSizeOpName[i])
			{
				hebing_flag=1;
			}	
		}
		if(hebing_flag==0)
		{
			split_index.push_back(it_layerIndex_opInfo->first);
		}
		it_layerIndex_opInfo++;
	}
	//hebing op to layers and build map outblobname and indexId 
	std::map<std::string,int> name2index;
	int blob_id_index=0;
	//std::cout<<split_index.size()<<std::endl;
	for(int i=1;i<split_index.size();i++)
	{
		int pre_split_index=split_index[i-1];
		int cur_split_index=split_index[i];
		std::vector<StOpInfo> hebing_opInfo;
		StOpInfo tempOpInfo;
		int last_layerIndex=0;
		for(int j=pre_split_index;j<cur_split_index;j++)
		{
			if(layerIndex_opInfo.count(j))//layerIndex bu lianxu
			{
				last_layerIndex=j;
				tempOpInfo=layerIndex_opInfo[j][0];
				hebing_opInfo.push_back(tempOpInfo);
				//std::cout<<tempName<<std::endl;
			}
			
		}
		hebing_layerIndex_opInfo[hebing_layerIndex]=hebing_opInfo;
		hebing_layerIndex_inputBlob[hebing_layerIndex]=layerIndex_inputBlob[pre_split_index];
		hebing_layerIndex_outputBlob[hebing_layerIndex]=layerIndex_outputBlob[last_layerIndex];
		//build map
		for(int io=0;io<hebing_layerIndex_outputBlob[hebing_layerIndex].size();io++)	
		{	
			std::string outputBlobname=hebing_layerIndex_outputBlob[hebing_layerIndex][io].blobName;
			name2index[outputBlobname]=blob_id_index;
			blob_id_index++;
			
		}
		hebing_layerIndex++;
		//zui hou yi ge xie ru
		if(i==split_index.size()-1)
		{
			hebing_opInfo.clear();
			//std::cout<<"manzu"<<std::endl;
			//std::cout<<cur_split_index<<","<<layerIndex_opInfo.size()<<std::endl;
			for(int j=cur_split_index;j<MAX_LAYERNUM;j++)
			{
				if(layerIndex_opInfo.count(j))//layerIndex bu lianxu
				{
					last_layerIndex=j;
					tempOpInfo=layerIndex_opInfo[j][0];
					hebing_opInfo.push_back(tempOpInfo);
					//std::cout<<tempName<<std::endl;
				}
				
			}
			hebing_layerIndex_opInfo[hebing_layerIndex]=hebing_opInfo;
			hebing_layerIndex_inputBlob[hebing_layerIndex]=layerIndex_inputBlob[cur_split_index];
			hebing_layerIndex_outputBlob[hebing_layerIndex]=layerIndex_outputBlob[last_layerIndex];
			//build map
			for(int io=0;io<hebing_layerIndex_outputBlob[hebing_layerIndex].size();io++)	
			{	
				std::string outputBlobname=hebing_layerIndex_outputBlob[hebing_layerIndex][io].blobName;
				name2index[outputBlobname]=blob_id_index;
				blob_id_index++;
				
			}
			hebing_layerIndex++;
		}
		
	}
	//op inputname and outputname to index int   not string
	it_layerIndex_inputBlob=hebing_layerIndex_inputBlob.begin();
	it_layerIndex_outputBlob=hebing_layerIndex_outputBlob.begin();
	while(it_layerIndex_inputBlob!=hebing_layerIndex_inputBlob.end()
		&&it_layerIndex_outputBlob!=hebing_layerIndex_outputBlob.end())
	{
		//add inputblobinfo blobID
		std::vector<StBlobInfo> blobInfo=it_layerIndex_inputBlob->second;
		for(int i=0;i<blobInfo.size();i++)
		{
			it_layerIndex_inputBlob->second[i].id_index=-1;
			if(name2index.count(blobInfo[i].blobName))
			{
				it_layerIndex_inputBlob->second[i].id_index=name2index[blobInfo[i].blobName];
			}
		}
		//add outputblobinfo blobID
		blobInfo=it_layerIndex_outputBlob->second;
		for(int i=0;i<blobInfo.size();i++)
		{
			it_layerIndex_outputBlob->second[i].id_index=-1;
			if(name2index.count(blobInfo[i].blobName))
			{
				it_layerIndex_outputBlob->second[i].id_index=name2index[blobInfo[i].blobName];
			}
		}
		//printf
		hebing_layerIndex=it_layerIndex_inputBlob->first;
		std::cout<<hebing_layerIndex<<"  ";
		for(int k=0;k<hebing_layerIndex_opInfo[hebing_layerIndex].size();k++)
		{
			std::cout<<hebing_layerIndex_opInfo[hebing_layerIndex][k].opName<<"  ";
			std::cout<<hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType<<"  ";
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpConv)
			{
				StConvParam *pstNetConvparam;
				pstNetConvparam=(StConvParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<"channel:"<<pstNetConvparam->iKernelChannel<<",bias_term:"<<pstNetConvparam->bias_term
						 <<",pad:"<<pstNetConvparam->ipad_left<<","<<pstNetConvparam->ipad_top<<","<<pstNetConvparam->ipad_right<<","<<pstNetConvparam->ipad_bottom
						 <<",kernel:"<<pstNetConvparam->ikernel_h<<","<<pstNetConvparam->ikernel_w
						 <<",stride:"<<pstNetConvparam->istride_h<<","<<pstNetConvparam->istride_w<<"  ";
			}
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpBn)
			{
				StBnParam *pstNetBnparam;
				pstNetBnparam=(StBnParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<pstNetBnparam->fEps<<"  ";
			}
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpScale)
			{
				StScaleParam *pstNetScaleparam;
				pstNetScaleparam=(StScaleParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<pstNetScaleparam->bias_term<<"  ";
			}
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpPool)
			{
				StPoolParam *pstNetPoolparam;
				pstNetPoolparam=(StPoolParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<"Pool method:"<<pstNetPoolparam->enPoolMethod
						 <<",pad:"<<pstNetPoolparam->ipad_left<<","<<pstNetPoolparam->ipad_top<<","<<pstNetPoolparam->ipad_right<<","<<pstNetPoolparam->ipad_bottom
						 <<",kernel:"<<pstNetPoolparam->ikernel_h<<","<<pstNetPoolparam->ikernel_w
						 <<",stride:"<<pstNetPoolparam->istride_h<<","<<pstNetPoolparam->istride_w<<"  ";
			}
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpEltwise)
			{
				StEltwiseParam *pstNetEltwiseparam;
				pstNetEltwiseparam=(StEltwiseParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<"Eltwise method:"<<pstNetEltwiseparam->enEltwiseMethod<<"  ";
			}
			if(hebing_layerIndex_opInfo[hebing_layerIndex][k].enfpgaOpType==OpFc)
			{
				StFcParam *pstNetFcparam;
				pstNetFcparam=(StFcParam*)(hebing_layerIndex_opInfo[hebing_layerIndex][k].pvOpParam);
				std::cout<<"bias term:"<<pstNetFcparam->bias_term<<",outchannel:"<<pstNetFcparam->iKernelChannel<<"  ";
			}	
		}
		for(int ii=0;ii<hebing_layerIndex_inputBlob[hebing_layerIndex].size();ii++)
		{
			std::cout<<hebing_layerIndex_inputBlob[hebing_layerIndex][ii].blobName<<"  ";
			std::cout<<hebing_layerIndex_inputBlob[hebing_layerIndex][ii].id_index<<"  ";
		}
		for(int io=0;io<hebing_layerIndex_outputBlob[hebing_layerIndex].size();io++)	
		{	
			std::cout<<hebing_layerIndex_outputBlob[hebing_layerIndex][io].blobName<<"  ";
			std::cout<<hebing_layerIndex_outputBlob[hebing_layerIndex][io].id_index<<"  ";
		}
		std::cout<<std::endl;
		it_layerIndex_inputBlob++;
		it_layerIndex_outputBlob++;
	}
	layerIndex_opInfo=hebing_layerIndex_opInfo;
	layerIndex_inputBlob=hebing_layerIndex_inputBlob;
	layerIndex_outputBlob=hebing_layerIndex_outputBlob;
	//std::copy(hebing_layerIndex_opInfo.begin(),hebing_layerIndex_opInfo.end(),std::inserter(layerIndex_opInfo,layerIndex_opInfo.begin()));
	
}

//comput inputsize and outputsize
int compute_layers_outputsize(StBlobShape stNetInput,std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob)
{
	std::map<int,StBlobShape> map_blobid_blobshape;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_inputBlob;
	std::map<int,std::vector<StBlobInfo>>::iterator  it_layerIndex_outputBlob;
	std::map<int,std::vector<StOpInfo>>::iterator it_layerIndex_opInfo;
	//build map blobid and blobsize
	it_layerIndex_inputBlob=layerIndex_inputBlob.begin();
	while(it_layerIndex_inputBlob!=layerIndex_inputBlob.end())
	{
		std::vector<StBlobInfo> vc_blobinfo=it_layerIndex_inputBlob->second;
		for(int i=0;i<vc_blobinfo.size();i++)
		{
			if(vc_blobinfo[i].id_index==-1)
			{
				memcpy(&vc_blobinfo[i].blobshape,&stNetInput,sizeof(StBlobShape));
			}
			map_blobid_blobshape[vc_blobinfo[i].id_index]=vc_blobinfo[i].blobshape;
		}
		it_layerIndex_inputBlob++;
	}
	//compute layer inputsize and outputsize
	it_layerIndex_opInfo=layerIndex_opInfo.begin();
	it_layerIndex_inputBlob=layerIndex_inputBlob.begin();
	it_layerIndex_outputBlob=layerIndex_outputBlob.begin();
	while(it_layerIndex_inputBlob!=layerIndex_inputBlob.end()
		&&it_layerIndex_outputBlob!=layerIndex_outputBlob.end()
		&&it_layerIndex_opInfo!=layerIndex_opInfo.end())
	{
		StOpInfo stopInfo=it_layerIndex_opInfo->second[0];
		std::vector<StBlobInfo> vc_inputblobinfo=it_layerIndex_inputBlob->second;
		std::vector<StBlobShape> vc_inputblobshape;
		std::vector<StBlobShape> vc_outputblobshape;
		//from map_blobid_blobshape find inputid->inputshape
		//std::cout<<stopInfo.opName<<"  ";
		for(int i=0;i<vc_inputblobinfo.size();i++)
		{
			int blob_id_index=vc_inputblobinfo[i].id_index;
			//std::cout<<"shape:"<<map_blobid_blobshape[blob_id_index].N<<" ";
			//std::cout<<map_blobid_blobshape[blob_id_index].C<<" ";
			//std::cout<<map_blobid_blobshape[blob_id_index].H<<" ";
			//std::cout<<map_blobid_blobshape[blob_id_index].W<<" ";
			vc_inputblobinfo[i].blobshape.N=1;
			vc_inputblobinfo[i].blobshape.C=map_blobid_blobshape[blob_id_index].C;
			vc_inputblobinfo[i].blobshape.H=map_blobid_blobshape[blob_id_index].H;
			vc_inputblobinfo[i].blobshape.W=map_blobid_blobshape[blob_id_index].W;
			vc_inputblobshape.push_back(map_blobid_blobshape[blob_id_index]);
		}
		//std::cout<<std::endl;
		it_layerIndex_inputBlob->second=vc_inputblobinfo;
		compute_op_outputsize(stopInfo,vc_inputblobshape,vc_outputblobshape);
		//update map_blobid_blobshape's blobshape
		std::vector<StBlobInfo> vc_outputblobinfo=it_layerIndex_outputBlob->second;
		if(vc_outputblobinfo.size()!=vc_outputblobshape.size())
		{
			std::cout<<"compute shape size error "<<std::endl;
			return -1;
		}
		else
		{
			for(int i=0;i<vc_outputblobshape.size();i++)
			{
				memcpy(&vc_outputblobinfo[i].blobshape,&vc_outputblobshape[i],sizeof(StBlobShape));
				map_blobid_blobshape[vc_outputblobinfo[i].id_index]=vc_outputblobinfo[i].blobshape;
			}
		}
		it_layerIndex_outputBlob->second=vc_outputblobinfo;
		it_layerIndex_inputBlob++;
		it_layerIndex_outputBlob++;
		it_layerIndex_opInfo++;
	}
	
}

void compute_op_outputsize(StOpInfo stopInfo,std::vector<StBlobShape> &vc_inputblobshape,std::vector<StBlobShape> &vc_outputblobshape)
{
	StBlobShape bloboutshape;
	bloboutshape.N=vc_inputblobshape[0].N;
	if(stopInfo.enfpgaOpType==OpConv)
	{
		if(vc_inputblobshape.size()>1)
		{
			std::cout<<"conv only one input"<<std::endl;
		}
		else
		{
			StConvParam *pstConvParam=(StConvParam *)stopInfo.pvOpParam;
			bloboutshape.H = (vc_inputblobshape[0].H + 2 * pstConvParam->ipad_top - (pstConvParam->idilation_size * (pstConvParam->ikernel_h - 1) + 1)) / pstConvParam->istride_h + 1;
			bloboutshape.W = (vc_inputblobshape[0].W + 2 * pstConvParam->ipad_left - (pstConvParam->idilation_size * (pstConvParam->ikernel_w - 1) + 1)) / pstConvParam->istride_w + 1;
			//std::cout<<pstConvParam->idilation_size<<"hhhhhhhh"<<pstConvParam->istride_w<<std::endl;
			bloboutshape.C=pstConvParam->iKernelChannel;
		}
		
		vc_outputblobshape.push_back(bloboutshape);
	}
	else if(stopInfo.enfpgaOpType==OpPool)
	{
		if(vc_inputblobshape.size()>1)
		{
			std::cout<<"pool only one input"<<std::endl;
		}
		else
		{
			StPoolParam *pstPoolParam=(StPoolParam *)stopInfo.pvOpParam;
			bloboutshape.H = (int)std::ceil(((float)(vc_inputblobshape[0].H+2 * pstPoolParam->ipad_top-pstPoolParam->ikernel_h))/pstPoolParam->istride_h)+1;
			bloboutshape.W = (int)std::ceil(((float)(vc_inputblobshape[0].W+2 * pstPoolParam->ipad_left-pstPoolParam->ikernel_w))/pstPoolParam->istride_w)+1;
			
			//std::cout<<pstConvParam->idilation_size<<"hhhhhhhh"<<pstConvParam->istride_w<<std::endl;
		}
		bloboutshape.C=vc_inputblobshape[0].C;
		vc_outputblobshape.push_back(bloboutshape);
	}
	else if(stopInfo.enfpgaOpType==OpFc)
	{
		if(vc_inputblobshape.size()>1 || vc_inputblobshape[0].H!=1 || vc_inputblobshape[0].W!=1)
		{
			std::cout<<"fc only one input"<<std::endl;
		}
		else
		{
			StFcParam *pstFcParam=(StFcParam *)stopInfo.pvOpParam;
			bloboutshape.H = 1;
			bloboutshape.W = 1;
			bloboutshape.C=pstFcParam->iKernelChannel;
			//std::cout<<pstConvParam->idilation_size<<"hhhhhhhh"<<pstConvParam->istride_w<<std::endl;
		}
		vc_outputblobshape.push_back(bloboutshape);
	}
	else if(stopInfo.enfpgaOpType==OpConcat)
	{
		bloboutshape.C = 0;
		for (int i = 0; i < vc_inputblobshape.size(); ++i)
		{
			bloboutshape.C += vc_inputblobshape[i].C;
			bloboutshape.H = vc_inputblobshape[i].H;
			bloboutshape.W = vc_inputblobshape[i].W;
			bloboutshape.N = vc_inputblobshape[i].N;
		}
		vc_outputblobshape.push_back(bloboutshape);
	}
	else
	{
		vc_outputblobshape.push_back(vc_inputblobshape[0]);
	}
	
}

void map_to_fpganetwork(std::string netName,std::map<int,std::vector<StOpInfo>> &layerIndex_opInfo,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_inputBlob,
				  std::map<int,std::vector<StBlobInfo>> &layerIndex_outputBlob)
{
	int imemsize=0;
	FILE *fnetwork=fopen("fpganetwork.bin","wb");
	StFpgaNetInfo stFpgaNetInfo;
	memset(&stFpgaNetInfo,0,sizeof(stFpgaNetInfo));
	//stFpgaNetInfo.archNetName= (char*)netName.c_str();
	netName.copy(stFpgaNetInfo.archNetName, netName.length(), 0);
	stFpgaNetInfo.archNetName[netName.length()]='\0';
	stFpgaNetInfo.iNetworkVersion=20190510;
	stFpgaNetInfo.iLayerNum=layerIndex_opInfo.size();
	StFpgaLayerInfo *pstLayerInfo=(StFpgaLayerInfo*)malloc(stFpgaNetInfo.iLayerNum*sizeof(StFpgaLayerInfo));
	fwrite(&stFpgaNetInfo,sizeof(stFpgaNetInfo),1,fnetwork);
	stFpgaNetInfo.pstFpgaLayerInfo=pstLayerInfo;
	
	imemsize+=sizeof(stFpgaNetInfo);
	std::cout<<sizeof(stFpgaNetInfo)<<std::endl;
	for(int i=0;i<stFpgaNetInfo.iLayerNum;i++)
	{
		if(i==71)
		{
			std::cout<<"fc before size:"<<imemsize;
		}
		memset(pstLayerInfo,0,sizeof(StFpgaLayerInfo));
		pstLayerInfo->iLayerId=i;
		pstLayerInfo->iOpNum=layerIndex_opInfo[i].size();
		pstLayerInfo->iBlobInputNum=layerIndex_inputBlob[i].size();
		pstLayerInfo->iBlobOutputNum=layerIndex_outputBlob[i].size();
		std::cout<<pstLayerInfo->iLayerId<<"  "
		         <<pstLayerInfo->iOpNum<<"  "
				 <<pstLayerInfo->iBlobInputNum<<"  "
				 <<pstLayerInfo->iBlobOutputNum<<"  ";
		//read input and output size and index
		for(int iblob=0;iblob<pstLayerInfo->iBlobInputNum;iblob++)
		{
			pstLayerInfo->ariInputBlobId[iblob]=layerIndex_inputBlob[i][iblob].id_index;
			memcpy(pstLayerInfo->arstInputfeatDim+iblob,&layerIndex_inputBlob[i][iblob].blobshape,sizeof(StBlobShape));
			std::cout<<pstLayerInfo->ariInputBlobId[iblob]<<"  "
					 <<pstLayerInfo->arstInputfeatDim[iblob].N<<"  "
					 <<pstLayerInfo->arstInputfeatDim[iblob].C<<"  "
					 <<pstLayerInfo->arstInputfeatDim[iblob].H<<"  "
					 <<pstLayerInfo->arstInputfeatDim[iblob].W<<"  ";
		}
		for(int iblob=0;iblob<pstLayerInfo->iBlobOutputNum;iblob++)
		{
			pstLayerInfo->ariOutputBlobId[iblob]=layerIndex_outputBlob[i][iblob].id_index;
			memcpy(pstLayerInfo->arstOutputfeatDim+iblob,&layerIndex_outputBlob[i][iblob].blobshape,sizeof(StBlobShape));
			std::cout<<pstLayerInfo->ariOutputBlobId[iblob]<<"  "
					 <<pstLayerInfo->arstOutputfeatDim[iblob].N<<"  "
					 <<pstLayerInfo->arstOutputfeatDim[iblob].C<<"  "
					 <<pstLayerInfo->arstOutputfeatDim[iblob].H<<"  "
					 <<pstLayerInfo->arstOutputfeatDim[iblob].W<<"  ";
		}
		//layer op info
		StFpgaOpInfo *pstOpInfo=(StFpgaOpInfo*)malloc(layerIndex_opInfo[i].size()*sizeof(StFpgaOpInfo));
		memset(pstOpInfo,0,sizeof(StFpgaOpInfo));
		fwrite(pstLayerInfo,sizeof(StFpgaLayerInfo),1,fnetwork);
		imemsize+=sizeof(StFpgaLayerInfo);
		pstLayerInfo->pstFpgaOpInfo=pstOpInfo;
		
		for(int iop=0;iop<layerIndex_opInfo[i].size();iop++)
		{
			pstOpInfo->enumOpType=layerIndex_opInfo[i][iop].enfpgaOpType;
			fwrite(pstOpInfo,sizeof(StFpgaOpInfo),1,fnetwork);
			pstOpInfo->pvOpInfo=layerIndex_opInfo[i][iop].pvOpParam;
			//std::cout<<pstOpInfo->enumOpType<<"enmu size:"<<sizeof(StFpgaOpInfo)<<"  ";
			
			imemsize+=sizeof(StFpgaOpInfo);
			if(pstOpInfo->enumOpType==OpConv)
			{
				StConvParam *pstNetConvparam=(StConvParam*)pstOpInfo->pvOpInfo;
				std::cout<<"channel:"<<pstNetConvparam->iKernelChannel<<",bias_term:"<<pstNetConvparam->bias_term
						 <<",pad:"<<pstNetConvparam->ipad_left<<","<<pstNetConvparam->ipad_top<<","<<pstNetConvparam->ipad_right<<","<<pstNetConvparam->ipad_bottom
						 <<",kernel:"<<pstNetConvparam->ikernel_h<<","<<pstNetConvparam->ikernel_w
						 <<",stride:"<<pstNetConvparam->istride_h<<","<<pstNetConvparam->istride_w<<"  ";
				fwrite(pstOpInfo->pvOpInfo,sizeof(StConvParam),1,fnetwork);
				imemsize+=sizeof(StConvParam);
			}
			if(pstOpInfo->enumOpType==OpBn)
			{
				StBnParam *pstNetBnparam;
				pstNetBnparam=(StBnParam*)(pstOpInfo->pvOpInfo);
				std::cout<<pstNetBnparam->fEps<<"  ";
				fwrite(pstOpInfo->pvOpInfo,sizeof(StBnParam),1,fnetwork);
				imemsize+=sizeof(StBnParam);
			}
			if(pstOpInfo->enumOpType==OpScale)
			{
				StScaleParam *pstNetScaleparam;
				pstNetScaleparam=(StScaleParam*)(pstOpInfo->pvOpInfo);
				fwrite(pstOpInfo->pvOpInfo,sizeof(StScaleParam),1,fnetwork);
				imemsize+=sizeof(StScaleParam);
			}
			if(pstOpInfo->enumOpType==OpPool)
			{
				StPoolParam *pstNetPoolparam;
				pstNetPoolparam=(StPoolParam*)(pstOpInfo->pvOpInfo);
				std::cout<<"Pool method:"<<pstNetPoolparam->enPoolMethod
						 <<",pad:"<<pstNetPoolparam->ipad_left<<","<<pstNetPoolparam->ipad_top<<","<<pstNetPoolparam->ipad_right<<","<<pstNetPoolparam->ipad_bottom
						 <<",kernel:"<<pstNetPoolparam->ikernel_h<<","<<pstNetPoolparam->ikernel_w
						 <<",stride:"<<pstNetPoolparam->istride_h<<","<<pstNetPoolparam->istride_w<<"  ";
				fwrite(pstOpInfo->pvOpInfo,sizeof(StPoolParam),1,fnetwork);
				imemsize+=sizeof(StPoolParam);
			}
			if(pstOpInfo->enumOpType==OpEltwise)
			{
				StEltwiseParam *pstNetEltwiseparam;
				pstNetEltwiseparam=(StEltwiseParam*)(pstOpInfo->pvOpInfo);
				std::cout<<"Eltwise method:"<<pstNetEltwiseparam->enEltwiseMethod<<"  ";
				fwrite(pstOpInfo->pvOpInfo,sizeof(StEltwiseParam),1,fnetwork);
				imemsize+=sizeof(StEltwiseParam);
			}
			if(pstOpInfo->enumOpType==OpFc)
			{
				StFcParam *pstNetFcparam;
				pstNetFcparam=(StFcParam*)(pstOpInfo->pvOpInfo);
				std::cout<<"bias term:"<<pstNetFcparam->bias_term<<",outchannel:"<<pstNetFcparam->iKernelChannel<<"  ";
				fwrite(pstOpInfo->pvOpInfo,sizeof(StFcParam),1,fnetwork);
				imemsize+=sizeof(StFcParam);
			}	
		}//end op
		free(pstOpInfo);
		pstOpInfo=NULL;
		std::cout<<std::endl;
	}//end layer
	free(pstLayerInfo);
	pstLayerInfo=NULL;
	fclose(fnetwork);
	std::cout<<"write size:"<<imemsize<<std::endl;
}
