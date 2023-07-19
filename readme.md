![](https://img.shields.io/badge/Language-python-brightgreen.svg)
# DDPM for unsupervised OCT denoising
### [SPIE 2022] Unsupervised denoising of retinal OCT with diffusion probabilistic model
---
- [x] The paper is available [here](https://arxiv.org/pdf/2201.11760.pdf)

### Introduction
Optical coherence tomography (OCT) is a prevalent non-invasive imaging method which provides high resolution volumetric visualization of retina. However, its inherent defect, the speckle noise, can seriously deteriorate the tissue visibility in OCT. Deep learning based approaches have been widely used for image restoration, but most of them require supervision from a noise-free reference image which is inaccessible for medical images. In this study, we present a diffusion probabilistic
model that is fully unsupervised to learn from noise instead of signal. A diffusion process is defined by adding a sequence of Gaussian noise to self-fused OCT b-scans. Then the reverse process of diffusion, modeled by a Markov chain, provides an adjustable level of denoising. Our experiment results demonstrate that our method
can significantly improve the image quality with a simple working pipeline and a small amount of training data.

The overall pipeline of the work is shown as following:
<p align="center">
  <img src="/assets/workflow.png" alt="drawing" width="650"/>
</p>

We first leverage the self-fusion method as a pre-processing step to create a relatively high SNR image as it is shown in **a.self-fusion**. Then we gradually add small Gaussian noise to the self-fused image as the diffusion process. The denoising process is realized by a deep model that learns the pattern of the noise. Detailed derivation is available in the paper.
>- The number of denoising step t is an extra hyperparameter. Then the model can denoise image with different noise level by adjusting t. In our experiment we show that the input with lower SNR needs more steps to reach the optimal visual effect. 

### Self-Fusion
Inherited from the joint label fusion, self-fusion regards b-scans in a small vicinity of a given target b-scan as ‘atlases’ because of their structural similarity. After registering the neighbors to the target b-scan, a pixel-wise weighted average of these ‘atlases’ will result in an image with high signal-to-noise ratio (SNR). The weight of each pixel is determined by a patch-wise similarity metric. The source paper is [**Self-fusion for OCT noise reduction**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8643350/), and a learning-based version is [**Retinal OCT Denoising with Pseudo-Multimodal Fusion Network**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9241435/). The label fusion software is availble under /label-fusion/, and an example bash file is provided (self_fusion.sh).

### Diffusion Probabilitic Model
The code is arranged as following:

       basic function and normalizing tools : util.py
              pre-processing and data loader: OCT_dataloader.py
    Gaussian diffusion and denoising process: DDPM_GuassianDiffusion.py
                        network architecture: DDPM_Net.py
                                    training: DDPM_main.py
                                     testing: DDPM_test.py

### Checkpoints
In the ckpts folder, the model used to denoise the retina OCT is provided. Note that the intensity should be normalized to range [1,3] for this model.

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
