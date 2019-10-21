# DNS-CP-Opt

## Introduction

Based on the improved channel pruning algorithm of the DNS(Dynamic Network Surgery for EfﬁcientDNNs),the number of input channels per layer is first pruned, and then fine-tuned on the pruned network to reduce the amount of computation while maintaining high precision. We implemented this algorithm using pytorch framework and FPGA platform. You can install pytorch and torchvision to use this demo.

## Installation

1. Python version shuold be 3.6+. 
2. Install PyTorch-0.4.1(the tested version) following the instructions on the official website. 
3. Clone this repository. 

## Run

Before running the demo,  you need to understand the basic steps of using pytorch to train the cnn network.

Use Imagenet to prune resnet50 (torchvision model): (you can train any model from torchvision on imagenet)

```
./run.sh
```

After two steps, the pruned model will be generated, for example:

```
$/pytorchmodel/finetune_best.pth.tar 
```
## Test
You can run pruned resnet50 model on caffe or download these model from https://1drv.ms/u/s!Am9Mk04MA_K1aVMJj0sTWk0x7Bw?e=KHYmfy.

## Reference

1. https://arxiv.org/abs/1608.04493
