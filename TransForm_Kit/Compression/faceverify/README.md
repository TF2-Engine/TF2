# FaceNet-FPGA

## Introduction

The sample program refers to the Facenet method and uses an 8-bit quantization method 
to achieve face detection and star-like face matching on the FPGA F10 card. 

## Usage

1. Install opencv and FPGA F10 card required software.

2. run:

   ```
   ./run.sh
   ```

   Parameter 1: Model Path;
   Parameter 2: Test Image Path;
   Parameter 3: Face Feature (binaryData) Library;
   Parameter 4: Face Image Library.

3. After running, you can get face detection results and similar faces.

## Tips:

1. You can download required dataset from https://1drv.ms/u/s!Am9Mk04MA_K1a6Q6xVMafKXMOgU?e=gdUIdz.
2. The compiled library is currently available and the code will be published later.
