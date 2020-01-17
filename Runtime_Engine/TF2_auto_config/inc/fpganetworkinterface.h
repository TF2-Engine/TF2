  
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
#ifndef __FPGANETWORKINTERFACE_H__
#define __FPGANETWORKINTERFACE_H__


#define Max_InputBlob_Num 10
#define Max_OutputBlob_Num 10
#define Max_Param_size 10240000 //10M

#include <stdio.h>
#include <fstream>
//fpga support op
#ifndef  EN_FPGAOP
#define  EN_FPGAOP
typedef enum  EN_fpgaop
{
	OpConv=1, //support
	OpFc,//support
	OpBn,//support
	OpScale,//support
	OpLrn,
	OpRelu,//support
	OpRelu6,
	OpPrelu,
	OpElu,
	OpSigmoid,
	OpPool,//support
	OpEltwise,//support
	OpConcat,
	OpFlatten,
	OpSoftmax,//support
	OpSlice,
	OpUnsupport
}EN_fpgaop;
#endif
//ActivationMethod
/*#ifndef EN_ACTIVATIONMETHOD
#define EN_ACTIVATIONMETHOD
typedef enum  EN_ActivationMethod
{
	Relu=0,
	Prelu,
	Elu,
	Sigmoid
}EN_ActivationMethod;
#endif*/
//PoolMethod
#ifndef EN_POOLMETHOD
#define EN_POOLMETHOD
typedef enum  EN_PoolMethod
{
	PoolMax=0,
	PoolAvg,
	PoolStochastic,
	PoolMean
	
}EN_PoolMethod;
#endif
//EltwiseMethod
#ifndef  EN_ELWISEMETHOD
#define  EN_ELWISEMETHOD
typedef enum  EN_EltwiseMethod
{
	EltwiseProd = 0,
    EltwiseSum,
    EltwiseMax
}EN_EltwiseMethod;
#endif
//conv param
#ifndef STCONVPARAM
#define STCONVPARAM
typedef struct StConvParam
{
	int  iKernelChannel;//channel
	bool bias_term; //bias flag
	int  ipad_left; //left pad num
	int  ipad_top;  //top pad num
	int  ipad_right;//right pad num
	int  ipad_bottom;//bottom pad num
	float fpadvalue;
	int  ikernel_h; //filter h
	int  ikernel_w; //filter_w
	int  istride_h; //the stride h
	int  istride_w; //the stride w
	int  idilation_size;//useless?
	//float *pfWeight; //weight
	//float *pfBias;   //fBias
}StConvParam;
#endif
//fc param
#ifndef STFCPARAM
#define STFCPARAM
typedef struct StFcParam
{
	int  iKernelChannel;//channel
	bool  bias_term; //bias flag
	//float *pfWeight; //weight
	//float *pfBias;   //bias
}StFcParam;
#endif
//bn param
#ifndef STBNPARAM
#define STBNPARAM
typedef struct StBnParam
{
	float  fEps;//torience 
	//int    iSize;//mean and varience scale size
	//float  *pfMean;//mean
	//float  *pfVarience;//varience
	//float  *pfScale;//size=1
}StBnParam;
#endif
//scale param
#ifndef STSCALEPARAM
#define STSCALEPARAM
typedef struct StScaleParam
{
	bool   bias_term;
}StScaleParam;
#endif
//lrn param
#ifndef STLRNPARAM
#define STLRNPARAM
typedef struct StLrnParam
{
	int    iLocal_size;
	float  fAlpha;
	float  fBeta;
}StLrnParam;
#endif
//activation param
#ifndef STELUPARAM
#define STELUPARAM
typedef struct StEluParam
{
	
	float fElualpha;//only for elu
}StEluParam;
#endif
//pool param
#ifndef STPOOLPARAM
#define STPOOLPARAM
typedef struct StPoolParam
{
	EN_PoolMethod enPoolMethod;
	int  ipad_left; //left pad num
	int  ipad_top;  //top pad num
	int  ipad_right;//right pad num
	int  ipad_bottom;//bottom pad num
	int  ikernel_h; //filter h
	int  ikernel_w; //filter_w
	int  istride_h; //the stride h
	int  istride_w; //the stride w
	bool global_pool;
}StPoolParam;
#endif
//eltwise param
#ifndef STELTWISEPARAM
#define STELTWISEPARAM
typedef struct StEltwiseParam
{
	EN_EltwiseMethod enEltwiseMethod;	
}StEltwiseParam;
#endif
//concat param
#ifndef STCONCATPARAM
#define STCONCATPARAM
typedef struct StConcatParam
{
	int iAxis; //By default, ConcatLayer concatenates blobs along the "channels" axis
}StConcatParam;
#endif
//Flatten param
#ifndef STFLATTENPARAM
#define STFLATTENPARAM
typedef struct StFlattenParam
{
	int iAxis;//which axis to flatten
}StFlattenParam;
#endif
//Slice param
#ifndef STSLICEPARAM
#define STSLICEPARAM
typedef struct StSliceParam
{
	int iAxis;//which axis to slice
	int iNumSlice;//slice out num
    int ariSliceSize[Max_OutputBlob_Num];//	
}StSliceParam;
#endif
//opinfo
#ifndef STFPGAOPINFO
#define STFPGAOPINFO
typedef struct StFpgaOpInfo
{
	EN_fpgaop enumOpType;
	void *pvOpInfo;
}StFpgaOpInfo;
#endif
#ifndef STBLOBSHAPE
#define STBLOBSHAPE
typedef struct StBlobShape
{
	int N;
	int C;
	int H;
	int W;
}StBlobShape;
#endif
//layer info
#ifndef STFPGALAYERINFO
#define STFPGALAYERINFO
typedef struct StFpgaLayerInfo
{
	int iLayerId;
	int iOpNum;
	int iBlobInputNum;//输入个数
	int iBlobOutputNum;//输出个数
	int ariInputBlobId[Max_InputBlob_Num];
	int ariOutputBlobId[Max_InputBlob_Num];;//由于单输出，因此outputblobid=layerid
	StBlobShape arstInputfeatDim[Max_InputBlob_Num]; //input channel  有问�?
	StBlobShape arstOutputfeatDim[Max_InputBlob_Num];//output channel 有问�?
	StFpgaOpInfo *pstFpgaOpInfo;//  conv->bn->relu  
}StFpgaLayerInfo;
#endif
//network info
#ifndef STFPGANETINFO
#define STFPGANETINFO
typedef struct StFpgaNetInfo
{
	int    iNetworkVersion;//network banbenhao
	char    archNetName[128];//netname
	int    iLayerNum;//total layer num
	StFpgaLayerInfo *pstFpgaLayerInfo;
}StFpgaNetInfo;

bool read_fpganetwork(std::string filename,StFpgaNetInfo *pstNetInfo);
#endif
#endif
