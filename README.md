# [ICME'2025] Expansive Supervision for Neural Radiance Fields
Official Implementation for ICME'2025 "Expansive Supervision for Neural Radiance Fields".
### [Paper](https://arxiv.org/pdf/2409.08056) | [Code](https://github.com/zwx-open/Expansive-Supervision) 

[Weixiang Zhang](https://weixiang-zhang.github.io/),
Wei Yao,
[Shuzhao Xie](https://shuzhaoxie.github.io/),
Shijia Ge,
[Chen Tang](https://www.chentang.cc/),
[Zhi Wang*](http://pages.mmlab.top/)<br>
Tsinghua University \
\*: Corresponding Author

This is the official PyTorch implementation of "Expansive Supervision for Neural Radiance Fields" (accepted by *ICME 2025*).

# Overview
<p align="center">
  <img src="./assets/es_nerf_teaser.png" style="width:90%;">
</p>

**Abstract.** Neural Radiance Field (NeRF) has achieved remarkable success in creating immersive media representations through its exceptional reconstruction capabilities. However, the computational demands of dense forward passes and volume rendering during training continue to challenge its real-world applications. In this paper, we introduce Expansive Supervision to reduce time and memory costs during NeRF training from the
perspective of partial ray selection for supervision. Specifically, we observe that training errors exhibit a long-tail distribution
correlated with image content. Based on this observation, our
method selectively renders a small but crucial subset of pixels
and expands their values to estimate errors across the entire
area for each iteration. Compared to conventional supervision,
our approach effectively bypasses redundant rendering processes,
resulting in substantial reductions in both time and memory
consumption. Experimental results demonstrate that integrating
Expansive Supervision within existing state-of-the-art acceleration frameworks achieves 52% memory savings and 16% time
savings while maintaining comparable visual quality


# Additional Related Research
Welcome to explore our related research. The source code for all works has been available.
- (*CVPR'2025*) EVOS: Efficient Implicit Neural Training via EVOlutionary Selector| 
[[paper]](https://arxiv.org/pdf/2412.10153) | 
[[project]](https://weixiang-zhang.github.io/proj-evos/) | 
[[code]](https://github.com/zwx-open/EVOS-INR) | 
- (*AAAI'2025*) Enhancing Implicit Neural Representations via Symmetric Power Transformation | 
[[paper]](https://arxiv.org/abs/2412.09213) | 
[[project]](https://weixiang-zhang.github.io/proj-symtrans/) | 
[[code]](https://github.com/zwx-open/Symmetric-Power-Transformation-INR) | 
- Recent Progress of Implicit Neural Representations | 
[[code]](https://github.com/zwx-open/Recent-Progress-of-INR)

# Citation
Please consider leaving a ⭐ and citing our paper if you find this project helpful:

```
@article{expansive-sup-nerf,
  title={Expansive supervision for neural radiance field},
  author={Zhang, Weixiang and Xie, Shuzhao and Ge, Shijia and Yao, Wei and Tang, Chen and Wang, Zhi},
  journal={arXiv preprint arXiv:2409.08056},
  year={2024}
}
```