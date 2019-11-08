# ModelConvert

## Introduction

This project can conver caffe and pytorch model to FPGA model. 

1. caffe2fpga:convert model from caffe to fpga

2. pytorch2caffe:convert model from pytorch to caffe 


## Installation

1. Python version should be 3.0+. 

2. Follow the instructions on the official website to install PyTorch.

3. Protobuf version should be 3.6+. 

## Run

1. caffe2fpga

```
First,you shold compile it:
cd caffe2fpga
mkdir build
cd build
cmake ..
make

second,you can convert your caffe model:
./build/src/caffe2fpg resnet50.prototxt resnet50.caffemodel
Then you can see fpganetwork.bin fpgamodel.bin in current path. 

```

2. pytorch2caffe:  

```
python main.py 

Then you can see xxx.prototxt xxx.caffemodel in Modefiles/xxx/ path.
```

3. pytorch2fpga

(1) convert pytorch model to caffe model with pytorch2caffe tool.

(2) convert caffe model to fpga model with caffe2fpga tool.



## References

1. https://github.com/Tencent/ncnn 

2. https://github.com/sksq96/pytorch-summary
