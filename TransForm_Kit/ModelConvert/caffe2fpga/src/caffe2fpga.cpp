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

#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <fstream>
#include <set>
#include <limits>
#include <map>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "caffe.pb.h"
#include "tools.h"
#include "fpganetworkinterface.h"


#include "caffe.pb.h"

#include <fstream>

#include <iostream>
int getParameterFormCaffeModel(std::string model_path, std::vector<float> & param)
{

	caffe::NetParameter netparam;

	std::ifstream caffemodel(model_path, std::ifstream::in | std::ifstream::binary);

	if (&caffemodel == NULL)
	{

		std::cout << "The ptr of caffemodel is NULL" << std::endl;

		return -1;

	}

	if (!caffemodel.is_open()) 
	{

		std::cout << "Can not open model" << std::endl;

		return -1;

	}
	std::ofstream output( "fpgamodel.bin", std::ios::out | std::ios::binary );
    if( ! output )
    {
        std::cout << "Open output file error!" << std::endl;
        return -1;
    }

    

	bool flag = netparam.ParseFromIstream(&caffemodel);

	int layer_size = netparam.layer_size();

	std::cout << "layer_size = " << layer_size << std::endl;

	caffe::LayerParameter* layerparam = NULL;

	for (size_t i = 0; i < layer_size; i++)

	{

		layerparam = netparam.mutable_layer(i);

		std::cout << "layer type:" << layerparam->type() << std::endl;

		int n = layerparam->mutable_blobs()->size();
		for(int iblob=0;iblob<n;iblob++)
		{
			const caffe::BlobProto& blob = layerparam->blobs(iblob);
			printf("%s layer: %s weight(%d)\n", layerparam->type().c_str(), layerparam->name().c_str(), blob.data_size());
			for (size_t j = 0; j < blob.data_size(); j++)

			{

				float weight = blob.data()[j];
				output.write((char*)&weight, sizeof(weight));
				//std::cout << "weight is:" << weight << std::endl;

				param.push_back(weight);

			}
		}

	}

	caffemodel.close();
	output.close();
	//google::protobuf::ShutdownProtobufLibrary();

	return 0;

}

static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int main(int argc, char** argv)
{
    if (!(argc == 3))
    {
        fprintf(stderr, "Usage: %s [caffeproto] [caffemodel] \n", argv[0]);
        return -1;
    }

    const char* caffeproto = argv[1];
    const char* caffemodel = argv[2];
	std::vector<float> fpgamodel;
    getParameterFormCaffeModel(caffemodel,fpgamodel);
    caffe::NetParameter proto;
    caffe::NetParameter net;

    // load
    bool s0 = read_proto_from_text(caffeproto, &proto);
    if (!s0)
    {
        fprintf(stderr, "read_proto_from_text failed\n");
        return -1;
    }

    bool s1 = read_proto_from_binary(caffemodel, &net);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }
    
    // rename mapping for identical bottom top style
    std::map<std::string, std::string> blob_name_decorated;

    // bottom blob reference
    std::map<std::string, int> bottom_reference;

    // global definition line
    // [layer count] [blob count]
    int layer_count = proto.layer_size();
	//get net name
	std::string netName;
	if(net.has_name())
	{
		netName=net.name();
	}
	else
	{
		netName="undefine";
	}
	//const caffe::BlobShape& bs = input_param.shape(0);
	//get net input
	int input_dim_size =proto.input_dim_size();
	std::cout<<netName<<","<<proto.input_dim_size()<<std::endl;
	StBlobShape stInputshape;
	if(input_dim_size==4)
	{
		stInputshape.N=proto.input_dim(0);
		stInputshape.C=proto.input_dim(1);
		stInputshape.H=proto.input_dim(2);
		stInputshape.W=proto.input_dim(3);
		for(int dimIndex=0;dimIndex<input_dim_size;dimIndex++)
		{
			std::cout<<proto.input_dim(dimIndex)<<",";
		}
	}
    else
    {
        int i=0;
        for (i=0; i<layer_count; i++)
        {
            const caffe::LayerParameter& layer = proto.layer(i);
            if (layer.type() == "Input")
            {
                const caffe::InputParameter& input_param = layer.input_param();
                const caffe::BlobShape& bs = input_param.shape(0);
                stInputshape.N=bs.dim(0);
                stInputshape.C=bs.dim(1);
                stInputshape.H=bs.dim(2);
                stInputshape.W=bs.dim(3);
                break;
            }
        }
        if(i==layer_count)
        {
            std::cout<<"can't find input shape"<<std::endl;
            return -1;
        }
    }
	std::cout<<std::endl;
    std::set<std::string> blob_names;
	int hiddenLayerNum=0;
	int maxOpParamSize=10240000;//10M
	int cur_memsize=0;
	void *pcPamraMemory=(void*)malloc(maxOpParamSize*sizeof(char));
	memset(pcPamraMemory,0,maxOpParamSize*sizeof(char));
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);
		
        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(blob_name) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[blob_name];
            }

            blob_names.insert(blob_name);

            if (bottom_reference.find(blob_name) == bottom_reference.end())
            {
                bottom_reference[blob_name] = 1;
            }
            else
            {
                bottom_reference[blob_name] = bottom_reference[blob_name] + 1;
            }
        }

        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;
            blob_names.insert(blob_name);
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                blob_names.insert(blob_name);
            }
        }
    }
    // remove bottom_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = bottom_reference.begin();
    while (it != bottom_reference.end())
    {
        if (it->second == 1)
        {
            bottom_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }
    // populate
    blob_name_decorated.clear();
    int internal_split = 0;
    std::map<int,std::vector<StOpInfo>> layerIndex_opInfo;
	std::map<int,std::vector<StBlobInfo>>layerIndex_inputBlob;
	std::map<int,std::vector<StBlobInfo>>layerIndex_outputBlob;
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);
	    std::vector<StOpInfo> opInfo;
		StOpInfo tempOpInfo;
		if (layer.type() == "BN")
        {
			std::cout<<"bnnnnnnnnn"<<std::endl;
			tempOpInfo.opName="Scale";
			tempOpInfo.enfpgaOpType=OpScale;
        }
        else if (layer.type() == "Convolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1)
			{
				tempOpInfo.opName="ConvolutionDepthWise";
				tempOpInfo.enfpgaOpType=OpUnsupport;
				//opInfo.push_back(tempOpInfo);
			}	
            else
			{  
				tempOpInfo.opName="Convolution";
				tempOpInfo.enfpgaOpType=OpConv;
				//opInfo.push_back(tempOpInfo);
			}
        }
        else if (layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
        {
            
			tempOpInfo.opName="ConvolutionDepthWise";
			tempOpInfo.enfpgaOpType=OpUnsupport;
			//opInfo.push_back(tempOpInfo);
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1)
			{
               
				tempOpInfo.opName="ConvolutionDepthWise";
				tempOpInfo.enfpgaOpType=OpUnsupport;
				//opInfo.push_back(tempOpInfo);
			}
            else
			{
                
				tempOpInfo.opName="Deconvolution";
				tempOpInfo.enfpgaOpType=OpUnsupport;
				//opInfo.push_back(tempOpInfo);
			}
        }
        else if (layer.type() == "ReLU6")
        {
            
			tempOpInfo.opName="ReLU6";
			tempOpInfo.enfpgaOpType=OpRelu6;
			//opInfo.push_back(tempOpInfo);
        }
        else
        {
			tempOpInfo.opName=layer.type().c_str();
			if(layer.type()=="BatchNorm")
			{
				tempOpInfo.enfpgaOpType=OpBn;
			}
			else if(layer.type()=="Scale")
			{
				tempOpInfo.enfpgaOpType=OpScale;
			}
			else if(layer.type()=="InnerProduct")
			{
				tempOpInfo.enfpgaOpType=OpFc;
			}
			else if(layer.type()=="ReLU")
			{
				tempOpInfo.enfpgaOpType=OpRelu;
			}
			else if(layer.type()=="Eltwise")
			{
				tempOpInfo.enfpgaOpType=OpEltwise;
			}
			else if(layer.type()=="Pooling")
			{
				tempOpInfo.enfpgaOpType=OpPool;
			}
			else if(layer.type()=="Softmax")
			{
				tempOpInfo.enfpgaOpType=OpSoftmax;
			}
            else if(layer.type()=="Input")
            {
                continue;
            }
			else 
			{
				tempOpInfo.enfpgaOpType=OpUnsupport;
			}
			//opInfo.push_back(tempOpInfo);
        }
        //layerIndex_opInfo[i+hiddenLayerNum]=opInfo;
		std::vector<StBlobInfo> inputblobInfo;
		StBlobInfo stInputBlobInfo;
        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(layer.bottom(j)) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[layer.bottom(j)];
            }

            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refidx = bottom_reference[blob_name] - 1;
                bottom_reference[blob_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                blob_name = blob_name + splitsuffix;
            }
			stInputBlobInfo.blobName=blob_name.c_str();
			inputblobInfo.push_back(stInputBlobInfo);
            
        }
		layerIndex_inputBlob[i+hiddenLayerNum]=inputblobInfo;
        // decorated
		//out blob info 
		std::vector<StBlobInfo> outblobInfo;
		StBlobInfo stOutBlobInfo;
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;
			stOutBlobInfo.blobName=blob_name.c_str();
			outblobInfo.push_back(stOutBlobInfo);
			layerIndex_outputBlob[i+hiddenLayerNum]=outblobInfo;
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
				stOutBlobInfo.blobName=blob_name.c_str();
				outblobInfo.push_back(stOutBlobInfo);
				layerIndex_outputBlob[i+hiddenLayerNum]=outblobInfo;
            }
        }
		
        // find blob binary by layer name
        int netidx;
        for (netidx=0; netidx<net.layer_size(); netidx++)
        {
            if (net.layer(netidx).name() == layer.name())
            {
                break;
            }
        }

        // layer specific params
        if (layer.type() == "BatchNorm")
        {
            const caffe::BatchNormParameter& batch_norm_param = layer.batch_norm_param();
            float eps = batch_norm_param.eps();
            StBnParam *pstNetBnparam;
			pstNetBnparam=(StBnParam*)(pcPamraMemory+cur_memsize);
			pstNetBnparam->fEps=eps;
			tempOpInfo.pvOpParam=(StBnParam*)pstNetBnparam;
			cur_memsize+=sizeof(StBnParam);
            //std::vector<float> ones(mean_blob.data_size(), 1.f);
            //fwrite(ones.data(), sizeof(float), ones.size(), bp);// slope
        }
        else if (layer.type() == "Concat")
        {
            const caffe::ConcatParameter& concat_param = layer.concat_param();
            int dim = concat_param.axis() - 1;
			StConcatParam *pstNetConcatparam;
			pstNetConcatparam=(StConcatParam*)(pcPamraMemory+cur_memsize);
			pstNetConcatparam->iAxis=dim;
			tempOpInfo.enfpgaOpType=OpConcat;
			tempOpInfo.pvOpParam=(StConcatParam*)pstNetConcatparam;
			cur_memsize+=sizeof(StConcatParam);
            
        }
        else if (layer.type() == "Convolution" || layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            
			StConvParam *pstNetConvparam;
			pstNetConvparam=(StConvParam*)(pcPamraMemory+cur_memsize);
			cur_memsize+=sizeof(StConvParam);
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                
				pstNetConvparam->ikernel_w=convolution_param.kernel_w();
				pstNetConvparam->ikernel_h=convolution_param.kernel_h();
				tempOpInfo.pvOpParam=pstNetConvparam;
            }
            else
            {
                
				pstNetConvparam->ikernel_h=convolution_param.kernel_size(0);//h w yizhi
				pstNetConvparam->ikernel_w=convolution_param.kernel_size(0);
				tempOpInfo.pvOpParam=pstNetConvparam;
            }
            
            pstNetConvparam->idilation_size=convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1;
			tempOpInfo.pvOpParam=pstNetConvparam;
			if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                
				pstNetConvparam->istride_h=convolution_param.stride_h();//h w yizhi
				pstNetConvparam->istride_w=convolution_param.stride_w();
				tempOpInfo.pvOpParam=pstNetConvparam;
            }
            else
            {
                
				pstNetConvparam->istride_h=convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;//h w yizhi
				pstNetConvparam->istride_w=convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;
				tempOpInfo.pvOpParam=pstNetConvparam;
            }
            if (convolution_param.has_pad_w() && convolution_param.has_pad_h())
            {
                
				pstNetConvparam->ipad_left=convolution_param.pad_w();//h w yizhi
				pstNetConvparam->ipad_top=convolution_param.pad_h();
				pstNetConvparam->ipad_right=convolution_param.pad_w();//h w yizhi
				pstNetConvparam->ipad_bottom=convolution_param.pad_h();
				tempOpInfo.pvOpParam=pstNetConvparam;
            }
            else
            {
                
				pstNetConvparam->ipad_left=convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;//h w yizhi
				pstNetConvparam->ipad_top=convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
				pstNetConvparam->ipad_right=convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;//h w yizhi
				pstNetConvparam->ipad_bottom=convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
            }
            
			pstNetConvparam->bias_term=convolution_param.bias_term();
			pstNetConvparam->iKernelChannel=convolution_param.num_output();
            
            int num_group = 1;
            if (layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
            {
                num_group = convolution_param.num_output();
            }
            else
            {
                num_group = convolution_param.group();
            }
        }
        else if (layer.type() == "Crop")
        {
            const caffe::CropParameter& crop_param = layer.crop_param();
            int num_offset = crop_param.offset_size();
            if (num_offset == 1)
            {
                int offset = crop_param.offset(0);
                int axis = crop_param.axis();
            }
            else if (num_offset == 2)
            {
                int woffset = crop_param.offset(1);
                int hoffset = crop_param.offset(0);
                
            }
            else if (num_offset == 3)
            {
                int woffset = crop_param.offset(2);
                int hoffset = crop_param.offset(1);
                int coffset = crop_param.offset(0);
                
            }
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            int group = convolution_param.group();
        }
        else if (layer.type() == "DetectionOutput")
        {
            const caffe::DetectionOutputParameter& detection_output_param = layer.detection_output_param();
            const caffe::NonMaximumSuppressionParameter& nms_param = detection_output_param.nms_param();
        }
        else if (layer.type() == "Dropout")
        {
            const caffe::DropoutParameter& dropout_param = layer.dropout_param();
        }
        else if (layer.type() == "Eltwise")
        {
            const caffe::EltwiseParameter& eltwise_param = layer.eltwise_param();
            int coeff_size = eltwise_param.coeff_size();
			StEltwiseParam *pstNetEltwiseparam;
			pstNetEltwiseparam=(StEltwiseParam*)(pcPamraMemory+cur_memsize);
			pstNetEltwiseparam->enEltwiseMethod=(EN_EltwiseMethod)(int)eltwise_param.operation();
			tempOpInfo.pvOpParam=(StEltwiseParam*)pstNetEltwiseparam;
			cur_memsize+=sizeof(StEltwiseParam);
        }
        else if (layer.type() == "ELU")
        {
            const caffe::ELUParameter& elu_param = layer.elu_param();
            
        }
        else if (layer.type() == "InnerProduct")
        {
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
            
			StFcParam *pstNetFcparam;
			pstNetFcparam=(StFcParam*)(pcPamraMemory+cur_memsize);
			pstNetFcparam->iKernelChannel=inner_product_param.num_output();
			pstNetFcparam->bias_term=inner_product_param.bias_term();
			tempOpInfo.pvOpParam=(StFcParam*)pstNetFcparam;
			cur_memsize+=sizeof(StFcParam);
        }
        else if (layer.type() == "Input")
        {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
        }
        else if (layer.type() == "LRN")
        {
            const caffe::LRNParameter& lrn_param = layer.lrn_param();
        }
        else if (layer.type() == "Normalize")
        {
            const caffe::NormalizeParameter& norm_param = layer.norm_param();
        }
        else if (layer.type() == "Pooling")
        {
            const caffe::PoolingParameter& pooling_param = layer.pooling_param();
			StPoolParam *pstNetPoolparam;
			pstNetPoolparam=(StPoolParam*)(pcPamraMemory+cur_memsize);
			pstNetPoolparam->enPoolMethod=(EN_PoolMethod)(int)pooling_param.pool();
			tempOpInfo.pvOpParam=pstNetPoolparam;
			cur_memsize+=sizeof(StPoolParam);
            if (pooling_param.has_kernel_w() && pooling_param.has_kernel_h())
            {
                
				pstNetPoolparam->ikernel_w=pooling_param.kernel_w();
				pstNetPoolparam->ikernel_h=pooling_param.kernel_h();
            }
            else
            {
                
				pstNetPoolparam->ikernel_w=pooling_param.kernel_size();
				pstNetPoolparam->ikernel_h=pooling_param.kernel_size();
            }
            if (pooling_param.has_stride_w() && pooling_param.has_stride_h())
            {
                
				pstNetPoolparam->istride_w=pooling_param.stride_w();
				pstNetPoolparam->istride_h=pooling_param.stride_h();
            }
            else
            {
                
				pstNetPoolparam->istride_w=pooling_param.stride();
				pstNetPoolparam->istride_h=pooling_param.stride();
            }
            if (pooling_param.has_pad_w() && pooling_param.has_pad_h())
            {
                
				pstNetPoolparam->ipad_left=pooling_param.pad_w();
				pstNetPoolparam->ipad_top=pooling_param.pad_h();
				pstNetPoolparam->ipad_right=pooling_param.pad_w();
				pstNetPoolparam->ipad_bottom=pooling_param.pad_h();
            }
            else
            {
                
				pstNetPoolparam->ipad_left=pooling_param.pad();
				pstNetPoolparam->ipad_top=pooling_param.pad();
				pstNetPoolparam->ipad_right=pooling_param.pad();
				pstNetPoolparam->ipad_bottom=pooling_param.pad();
            }
            
			pstNetPoolparam->global_pool=pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0;
        }
        else if (layer.type() == "PReLU")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
        }
        else if (layer.type() == "ReLU")
        {
            const caffe::ReLUParameter& relu_param = layer.relu_param();
            if (relu_param.has_negative_slope())
            {
				std::cout<<"relu unspport has_negative_slope"<<std::endl;
                
            }
        }
        else if (layer.type() == "Scale")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::ScaleParameter& scale_param = layer.scale_param();
            bool scale_weight = scale_param.bias_term() ? (binlayer.blobs_size() == 2) : (binlayer.blobs_size() == 1);
            if (scale_weight)
            {
                const caffe::BlobProto& weight_blob = binlayer.blobs(0);
                
            }
			StScaleParam *pstNetScaleparam;
			pstNetScaleparam=(StScaleParam*)(pcPamraMemory+cur_memsize);
			pstNetScaleparam->bias_term=scale_param.bias_term();
			tempOpInfo.pvOpParam=pstNetScaleparam;
			cur_memsize+=sizeof(StScaleParam);
        }
        else if (layer.type() == "Softmax")
        {
            const caffe::SoftmaxParameter& softmax_param = layer.softmax_param();
            int dim = softmax_param.axis() - 1;
        }
        else if (layer.type() == "Threshold")
        {
            const caffe::ThresholdParameter& threshold_param = layer.threshold_param();
        }
		opInfo.push_back(tempOpInfo);				
		layerIndex_opInfo[i+hiddenLayerNum]=opInfo;
        // add split layer if top reference larger than one
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = blob_name_decorated[layer.top(0)];
            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refcount = bottom_reference[blob_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
					opInfo.clear();
					tempOpInfo.opName="Split";// split wei hiddenLayer so  two op
					opInfo.push_back(tempOpInfo);
					hiddenLayerNum++;
					layerIndex_opInfo[i+hiddenLayerNum]=opInfo;
					inputblobInfo.clear();
					outblobInfo.clear();
					stInputBlobInfo.blobName=blob_name.c_str();
					inputblobInfo.push_back(stInputBlobInfo);
                    for (int j=0; j<refcount; j++)
                    {
                        
						char splitsuffix[256];
						sprintf(splitsuffix, "_splitncnn_%d", j);
						std::string new_blob_name = blob_name + splitsuffix;
						stOutBlobInfo.blobName=new_blob_name.c_str();
						
						outblobInfo.push_back(stOutBlobInfo);
                    }
					layerIndex_inputBlob[i+hiddenLayerNum]=inputblobInfo;
					layerIndex_outputBlob[i+hiddenLayerNum]=outblobInfo;
                    internal_split++;
                }
            }
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                if (bottom_reference.find(blob_name) != bottom_reference.end())
                {
                    int refcount = bottom_reference[blob_name];
                    if (refcount > 1)
                    {
                        char splitname[256];
                        sprintf(splitname, "splitncnn_%d", internal_split);
						opInfo.clear();
						tempOpInfo.opName="Split";
						opInfo.push_back(tempOpInfo);
						hiddenLayerNum++;
						layerIndex_opInfo[i+hiddenLayerNum]=opInfo;
						inputblobInfo.clear();
						outblobInfo.clear();
						stInputBlobInfo.blobName=blob_name.c_str();
						
						inputblobInfo.push_back(stInputBlobInfo);
                        for (int j=0; j<refcount; j++)
                        {
                            
							char splitsuffix[256];
							sprintf(splitsuffix, "_splitncnn_%d", j);
							std::string new_blob_name = blob_name + splitsuffix;
							stOutBlobInfo.blobName=new_blob_name.c_str();
							
							outblobInfo.push_back(stOutBlobInfo);
                        }
                        
						layerIndex_inputBlob[i+hiddenLayerNum]=inputblobInfo;
						layerIndex_outputBlob[i+hiddenLayerNum]=outblobInfo;
                        internal_split++;
                    }
                }
            }
        }
		
    }//end layercount
	
	remove_split(layerIndex_opInfo,layerIndex_inputBlob,layerIndex_outputBlob);
	hebing_op(layerIndex_opInfo,layerIndex_inputBlob,layerIndex_outputBlob);
	compute_layers_outputsize(stInputshape,layerIndex_opInfo,layerIndex_inputBlob,layerIndex_outputBlob);
	map_to_fpganetwork(netName,layerIndex_opInfo,layerIndex_inputBlob,layerIndex_outputBlob);
	StFpgaNetInfo *pstNetInfo;
	pstNetInfo=(StFpgaNetInfo *)malloc(Max_Param_size*sizeof(char));
	read_fppanetwork("./fpganetwork.bin",pstNetInfo);
	free(pstNetInfo);
	pstNetInfo=NULL;
	
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
		std::cout<<it_layerIndex_opInfo->first<<"	";
		std::vector<StOpInfo> opInfo=it_layerIndex_opInfo->second;
		for(int i=0;i<opInfo.size();i++)
		{
			std::cout<<opInfo[i].opName<<"	";
		}
		std::vector<StBlobInfo> blobInfo;
		blobInfo=it_layerIndex_inputBlob->second;
		for(int i=0;i<blobInfo.size();i++)
		{
			std::cout<<blobInfo[i].blobName<<"	";
		}
		blobInfo=it_layerIndex_outputBlob->second;
		for(int i=0;i<blobInfo.size();i++)
		{
			std::cout<<blobInfo[i].blobName<<"	";
		}
		it_layerIndex_opInfo++;
		it_layerIndex_inputBlob++;
		it_layerIndex_outputBlob++;
		std::cout<<std::endl;
	}
	free(pcPamraMemory);
	pcPamraMemory=NULL;
    return 0;
}
