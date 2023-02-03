![](https://img.shields.io/badge/Language-python-brightgreen.svg)
# DDPM for unsupervised OCT denoising
### [SPIE 2022] Unsupervised denoising of retinal OCT with diffusion probabilistic model
---
- [x] The paper is available [here](https://arxiv.org/pdf/2201.11760.pdf)

### Introduction
Optical coherence tomography (OCT) is a prevalent non-invasive imaging method which provides high resolution
volumetric visualization of retina. However, its inherent defect, the speckle noise, can seriously deteriorate the
tissue visibility in OCT. Deep learning based approaches have been widely used for image restoration, but most
of them require supervision from a noise-free reference image which is inaccessible for medical images. In this study, we present a diffusion probabilistic
model that is fully unsupervised to learn from noise instead of signal. A diffusion process is defined by adding
a sequence of Gaussian noise to self-fused OCT b-scans. Then the reverse process of diffusion, modeled by a
Markov chain, provides an adjustable level of denoising. Our experiment results demonstrate that our method
can significantly improve the image quality with a simple working pipeline and a small amount of training data.

The overall pipeline of the work is shown as following:
<p align="center">
  <img src="/assets/workflow.png" alt="drawing" width="650"/>
</p>


Please cite our work:
```
  @inproceedings{hu2022unsupervised,
  title={Unsupervised denoising of retinal OCT with diffusion probabilistic model},
  author={Hu, Dewei and Tao, Yuankai K and Oguz, Ipek},
  booktitle={Medical Imaging 2022: Image Processing},
  volume={12032},
  pages={25--34},
  year={2022},
  organization={SPIE}
}
```
