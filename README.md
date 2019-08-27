# Inspur Deep Learning Inference Accelerator TF2

## TF2

TF2 is a deep learning inference accelerator based on FPGA computing platform and developed by Inspur AI & HPC. It is suitable for general deep learning neural network. TF2 can parse models form deep learning frameworks such as Pytorch, TensorFLow, and Caffe. Users can adapt the trained deep learning model to FPGA through compiling without any FPGA development work, so it can help users quickly implement  AI applications on FPGA. See the link for the paper A Deep Learning Inference Accelerator Based on Model Compression on FPGA.

The TF2 acceleration engine consists of two parts: Transforme Kit and Runtime Engine. The Transform Kit is a model optimization conversion tool, which has the functions of model compression, pruning and quantification, etc., which can minimize the size of the model and reduce the amount of calculation. Transform Kit can also perform computational node fusion, and integrate multiple computing nodes into one node to reduce the impact of data access bandwidth on computing performance. Runtime Engine can automatically convert the previously optimized model file into FPGA target running file by compiling.

![top](./imgs/top.png)

## TransForm Kit

The TransForm Kit mainly includes four parts: model compression, cropping, quantization, and operation node fusion. It can receive model training by frameworks such as Pytorch, TensorFlow, Caffe, etc. After compression, cropping, quantification and operation node fusion, an intermediate representation computing graph file is generated for use in TF2 Runtime Engine calculation. The compression and cropping algorithms can be selected according to the actual application.

### Compression

Model compression uses Inspur optimized Incremental Network Quantization(INQ) compression method to compress the deep neural network model data trained by Pytorch and other frameworks. It can compress 32-bit floating-point model data into 4-bit integer data model, making the actual model data size is reduced to 1/8 of the original, and the storage structure of the original model data is basically maintained. The real data of the compressed model is the power of 2 of the 4-bit data or 0. Four 4-bit data is stored as 1 short type data. See link for detailed data formats. (After the reference paper)  The accuracy of the typical CNN neural network before and after compression is shown in the following table.

| NetWork    | Top1   | Top5   | Top1(Compressed) | Top5 (Compressed) |
| ---------- | ------ | ------ | ---------------- | ----------------- |
| Alexnet    | 0.5676 | 0.7990 | 0.5687           | 0.8000            |
| VGG16      | 0.6828 | 0.8827 | 0.7055           | 0.8994            |
| GoogLeNet  | 0.6889 | 0.8898 | 0.6857           | 0.8887            |
| ResNet50   | 0.7276 | 0.9101 | 0.7465           | 0.9248            |
| SqueezeNet | 0.5750 | 0.8030 | 0.5900           | 0.8040            |

| NetWork | Map    | Map(Compressed) |
| ------- | ------ | --------------- |
| SSD     | 0.7773 | 0.7757          |

### Pruning

TF2 Prune unit includes random pruning and channel pruning(code will be published later). The random pruning algorithm randomly prune the model data, the prune rate is high, but the model after pruning is a sparse model. Channel pruning is a kind of structured pruning, which can dynamically prune the channel. This method can directly reduce the number of channels and reduce the amount of calculation. The advantage of this method is that after the channel is pruned, the original precision is restored and the number of training times is small; the pruned model can be directly calculated based on the existing architecture TF2 architecture. The pruning rate of ResNet50 and the precision before and after pruning are shown in the table below. See the link for the pruned model. Pruning the model allows for a 1.6x speedup on the FPGA.

| Pruned Ratio | Top1   | Top5   | Top1-gap | Top5-gap |
| ------------ | ------ | ------ | -------- | -------- |
| 0%           | 0.7277 | 0.9109 | -        | -        |
| 50%          | 0.7289 | 0.9118 | 0.13% ↑  | 0.17% ↑  |
| 60%          | 0.7183 | 0.9079 | 0.93% ↓  | 0.22% ↓  |

### Quantization

Since the model is already 4-bit data after compressed in TF2, the quantization in TF2 is only the quantization of the feature map data. TF2 quantization tool can quantize normalized 32-bit single-precision floating-point feature map data to 8-bit integer, that is, quantized to -128 to 127. The main advantage of feature graph data quantization is that it can reduce the storage resource requirement of on-chip feature data by a quarter, and can reduce the logic calculation required for data processing, greatly improving the computing power of FPGA. The algorithm is described as follows:

1) Calculating the maximum value fmax of the absolute value of the feature map data of each channel of each convolution layer of the neural network;

2) Find Q according to the equation: 128 = power(2, Q) * fmax;

3) According to the above equation, the quantized data V = power(2, Q) * fv, where fv is the original single-precision floating-point data.

The calculation accuracy of deep learning neural network SqueezeNet and ResNet50 before and after quantization is shown in the following table.

| NetWork    | Top1   | Top5   | Top1(Quantized) | Top5(Quantized) |
| ---------- | ------ | ------ | --------------- | --------------- |
| Squeezenet | 0.5900 | 0.8040 | 0.5900          | 0.8010          |
| ResNet50   | 0.7145 | 0.9010 | 0.7120          | 0.9043          |

## RunTime Engine

The TF2 Runtime Engine is an FPGA intelligent runtime accelerator that automatically generates FPGA running files. It first parse the network structure file and generates the running parameter file required by the Runtinme Engine, and then recompiles FPGA code, which can automatically generates FPGA running file.

Since the Transform Kit compresses the model data into 4-bit data and the real model data is the power of 2 of the 4-bit data, the feature map data has also been quantized into 8-bit integers, so in the Runtime Engine, the multiplication of model data and the feature map data can be converted to shift and calculation, which eliminates the dependence of deep neural networks such as CNN on the DSP floating-point computing power of FPGA, greatly improves the performance of FPGA inference calculation and effectively reduces its actual running power consumption.

The Transform Kit FPGA top-level computing architecture is shown below. Multiple convolutional layers are executed serially on the FPGA. In order to reduce the impact of storage access on computing performance, the intermediate feature map data is stored on the chip as much as possible. The model data is read from the external DDR to the FPGA in real time during the calculation process, but its reading can be performed simultaneously with the calculation, that is, the reading time can be hidden by the calculation time. The TransForm Kit FPGA core computing architecture is shown below.

![block.png](./imgs/block.png)

Among them, the Filter Loader reads model data from the DDR to the chip. The Feature Loader reads the input picture and feature map data from the DDR to the on-chip. The Controller generates a control signal for the Scheduler. The Scheduler reads the Feature data according to the control signal, and sends data such as Feature, Filter, timing signal, and control signal to the PE for calculation. PE Array is the core computation unit of the entire computing architecture, performing Shift Accumulate (SAC) or Multiply Accumulate(MAC) calculations . The current version is SAC, which will be followed by MAC computing. There is a Filter Cache in the PE, which is used to store the Filter data to calculate the current output channel. The Adder adds the partial results of the MAC/SAC calculations to generate the final convolution calculation results. The number of PEs in a PE Array can be configured according to the structure of the neural network and the amount of FPGA resources. MAC/SAC can be calculated in 1D, 2D, or 3D, and can be configured according to actual conditions. The current version has 2D and 3D calculations. The vector length calculated by each MAC/SAC dimension can also be configured according to the specific application and FPGA computing resources. Networks such as ResNet50/SqueezeNet computiing performance list as follows.

| NetWork                   | Throughput(fps) |
| ------------------------- | --------------- |
| SqueezeNet                | 1485            |
| GoogLeNet                 | 306             |
| FaceNet(MTCNN+SqueezeNet) | 1020            |



## 开源RoadMap

#### 2019-Q2

Transform kit: Model comression and 8-bit quantization algorithm based on Pytorch.

Runtime Engine: CNN accelerator architecture based on OpenCL that uses SAC caculations.

Models: ResNet50, GoogLeNet, etc., can be tested directly based on the Inpur F10A board.

#### 2019-Q3

Transform Kit: Random pruning and channel pruning algorithm for CNN, automated model conversion tool.

Runtime Engine: An automatic model analysis tool and new CNN accelerator architecture for MAC caculations.

#### 2019-Q4

Transform Kit: More structural pruning algorithms and 4-bit quantization algorithm for CNN.

Runtime Engine: A general-purpose computing architecture for sparse models.

#### 2020-Q1

Transform kit: Any bit quantization algorithm for CNN networks.

Runtime Engine: Accelerator architecture that supports TransFormer computing.

#### 2020-Q2

Transform Kit: AutoML based pruning and quantization algorithm, NLP network optimization algorithm.

Runtime Engine: Support the NLP universal model and update the computing architecture continuously.

## Releases and Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, or add new algorithms about compression or pruning, please do so without any further discussion.

If you plan to update computing architecture of FPGA, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the architecture in a different direction than you might be aware of.

## Reference

1. Utku Aydonat, Shane O’Connell, Davor Capalija, Andrew C Ling, and Gordon RChiu. 2017. An Open deep learning accelerator on arria 10. In Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays. ACM, 55–64.
2. Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, and Yurong Chen. 2017. Incre-mental network quantization: Towards lossless cnns with low-precision weights.arXiv preprint arXiv:1702.03044 (2017).
