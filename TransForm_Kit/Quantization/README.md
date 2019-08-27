# CAQ: CNN Activation Quantization

## Introduction

A Pytorch implementation of CNN Activation Quantization method used for shift based FPGA CNN inference. The CAQ algorithm could be simply described as: 

1. Prepare the quantization dataset: select pictures from the dataset randomly and try to make the slected picture number equal to the classes number. For the imagenet, 1000 pictures should be selected. If you are not sure about the classes number, the slected pictures should not be less than 2% of the total pictures. 
2. Fp32 inference with the given weights: traverse the quantization dataset and get the maximum value of the activations. 
3. Quantize the activations by channel: caculate the mean value of each channel, if the activations absolute value is greater than the mean , then the values will be forced equal to the mean. The others values remain unchanged. 

For now, this implementation support the ResNet50, GoogLeNet, SqueezeNet, SSD and easy to modify for other CNN.

## Installation

1. Python version shuold be 3+. 
2. Install PyTorch follows the instructions on the official website. 
3.  Clone this repository. 

## Datasets

1. ImageNet.
2. VOC2007.
3. LFW, note that the LFW dataset should be aligned to (1,3,160,160) by alignment CNN like as mtcnn.
4. Your own dataset.

## Run

1. Inference and get the activations:


```
python feature_write.py 'net_name'
```

2. Quantization:  

```
python quantization.py 'net_name'
```

3. Modify for your own CNN
   i  Add to the CNN model to the models directory;
   ii Add your dataset preprocess and data loading method in model_loader.py and data_loader.py respectively;
   iii Assuming the coresponding dataset is ready, then repeat 1 and 2.

Note: for 1 and 2, the followed argument 'net_name' is 'resnet50' or  'googlenet' or 'ssd' or 'squeezenet', etc..

## References

1. https://github.com/amdegroot/ssd.pytorch 

2. https://github.com/sksq96/pytorch-summary
