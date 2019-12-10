# Winograd
  The winograd algorithm is often used for convolution acceleration. The TF2 Engine also implemented the CNN inference with winograd algorithm. Although the winograd could accelerate the inference, the computing resources have also increased. The convolution with winograd should be quantized. The differnce with the  shift convolution is the convolution with winograd could be quantized simply by a linear int8 quantization algorithm.
  The 1D winograd is Y = A^T(Gg)(B^Td), 2D winograd is Y = A^T(GgG^T)(B^TdB)A, where the G,B,and A is the filter, feature and product transformation matrix, respectively. 
  The quantization and inference with winograd can be described as: (1) quantize the FP32 weights to int8 with linear int8 quantization and get the quantized scale of weights (2) transform the weights with winograd transformation matrix (G,G^T) (3) get the feature quantized scale with linear int8 quantization (4) transform the input or features with transformation matrix (B,B^T) (5) multiply transformed weights and features element-wise (6) transform the element-wise product with transformation matrix (A,A^T).
## Quantization
1. To get the quantized scale and weights for the TF2 FPGA inference engine, the weights can be quantized and written by the weight_write.py.
2. Using 'python quantization.py winograd 'net name'' to quantize the features and get the quantized scale 
## Winograd transformation
The winograd transformation is different between the F(2,3), F(4,3) and F(5,3) implementation. The filter and feature could be transformed by filter_transfer.py and feature_transfer.py, respectively.
## The first layer
For some CNN model, for example the ResNet50 and GoogleNet, the first layer filter is 7x7 and it is not suitable for the winograd. One can  convert the 7x7 convolution to 3x3 convolution using the algorigthm as shown in conv7x7_2_3x3.py.
## Example
The example is a 1D int8 winograd inference simulation that implemented by python, it shows how to implement a inference with int8 winograd algorithm completely and one can easily modify it to implement your own CNN model.  
