# CAQ: CNN Activation Quantization

## Introduction

A PyTorch implementation of CAQ method that used for TF2 FPGA inference. The CAQ algorithm could be simply described as: 

1. Prepare the quantization dataset: Random sampling from the validation dataset and try to make the samples size equal to the amount of  classes. For the ImageNet, 1000 pictures should be selected. If you are not sure about the amount of classes, the selected quantization dataset size should not be less than 2% of the original dataset size. 

2. FP32 inference with the given weights: Traverse the quantization dataset and get the maximum values of the activations. 

3. Quantize the activations by channel: Calculate the means of the tensors of each channel. If the absolute values of activations are greater than the means, then the values will be forced to be equal to the means. The other values remain unchanged. 

Currently, this implementation supports ResNet50, GoogLeNet (Inception V1), SqueezeNet, SSD and it's easy to modify for other CNN.

## Installation

1. Python version should be 3.0+. 

2. Follow the instructions on the official website to install PyTorch.

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
