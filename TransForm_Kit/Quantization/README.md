# CAQ: CNN Activation Quantization

## Introduction

A Pytorch implementation of CAQ method that used for TF2 FPGA inference. The CAQ algorithm could be simply described as: 

1. Prepare the quantization dataset: random sampling from the validation dataset and try to make the number of samples equal to the classes number. For the ImageNet, 1000 pictures should be selected. If you are not sure about the classes number, then the slected quantization dataset should not be less than 2% of the original dataset. 

2. FP32 inference with the given weights: traverse the quantization dataset and get the maximum value of the activations. 

3. Quantize the activations by channel: caculate the mean value of each channel, if the activations absolute value is greater than the mean , then the values will be forced equal to the mean. The others values remain unchanged. 

For now, this implementation supports resnet50, googlenet, squeezenet, ssd and it's easy to modify for other CNN.

## Installation

1. Python version shuold be 3+. 

2. Follow the instructions on the official website to install Pytorch.

3. Clone this repository. 

## Datasets

1. ImageNet.

2. VOC2007.

3. LFW.

4. Your own dataset.

Note: the LFW dataset should be aligned to 3 x 160 x 160 (channel x height x weight) by alignment CNN like as MTCNN.

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

(1) Add the CNN model to the models directory.

(2) Add your dataset preprocess and data loading method in model_loader.py and data_loader.py respectively.

(3) Assuming the coresponding dataset is ready, then repeat 1 and 2.

Note: for 1 and 2, the followed argument 'net_name' is 'resnet50' or 'googlenet' or 'ssd' or 'squeezenet'.

## References

1. https://github.com/amdegroot/ssd.pytorch 

2. https://github.com/sksq96/pytorch-summary
