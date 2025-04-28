# NablaQDARTS
Source code of TMLR paper: ∇QDARTS: Quantization as an Elastic Dimension to Differentiable NAS

## Introduction
This work introduces ∆QDARTS, a novel approach that combines differentiable NAS with mixed-precision search. ∆QDARTS aims to identify the optimal mixed-precision neural architecture capable of achieving remarkable accuracy while operating with minimal computational requirements in a single shot, end-to-end differentiable framework obviating the need for proxies or pretraining.

## Folder Structure
#### PTQ - Post Training Quantization (Baseline)
Post-training quantization (PTQ) applied on the final trained architecture with fp32 precision (i.e., SubNet) discovered by PC-DARTS.
#### FBQAT - Fixed-bit Quantized Aware Training (Baseline)
Quantization aware training applied to the PC-DARTS full precision searched architecture using uniform, fixed-bit quantization policy 2- and 4-bit
#### QSubNet - Quantization on SubNet (Baseline)
Differentiable mixed-precision quantization technique (EdMIPS) applied on the output of search stage of PC-DARTS
####  𝚫QDARTS(QDARTS) - Proposed approach
End-to-end differentiable NAS joint with mixed-precision quantization search

Each of these folders have a ReadME with instructions for execution

### Pre-requisites for experiments with Imagenet
Most of the imagenet experiments use ffcv data loaders to reduce data bottlenecks. This requires the below steps

1) Clone https://github.com/libffcv/ffcv & https://github.com/libffcv/ffcv-imagenet

2) Run the below to create the conda environment with the necessary dependencies
```
conda create -y -n qdarts_ffcv_env python=3.10.8 cupy pkg-config libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
conda activate qdarts_ffcv_env
pip install ffcv
```

3) Generate ffcv ImageNet dataset
```
# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Starting in the root of the ffcv Git repo:
cd examples;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```
and use this saved image path as --trainfile and --valfile arguments to the training and evaluation scripts

## References
This repository makes use of the implementations of 

Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian, and Hongkai Xiong.
Pc-darts: Partial channel connections for memory-efficient architecture search. arXiv preprint
arXiv:1907.05737, 2019 (https://github.com/yuhuixu1993/PC-DARTS)

Zhaowei Cai and Nuno Vasconcelos. Rethinking differentiable search for mixed-precision
neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2349–2358, 2020. (https://github.com/zhaoweicai/EdMIPS)

Guillaume Leclerc, Andrew Ilyas, Logan Engstrom, Sung Min Park, Hadi Salman, and
Aleksander Madry. FFCV: Accelerating training by removing data bottlenecks, 2022. (https://github.com/libffcv/ffcv/)
