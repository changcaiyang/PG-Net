# PG-Net 
PyTorch implementation of PG-Net for TGRS'2023 paper "PG-Net: Progressive Guidance Network via Robust Contextual Embedding for Efficient Point Cloud Registration", by Jingtao Wang, Xin Liu, Luanyuan Dai, Jiayi Ma, Lifang Wei, Changcai Yang and Riqing Chen.
This paper focus on outlier rejection for 3D point clouds registration. If you find this project useful, please cite:
```
@article{10098825,
  title={PG-Net: Progressive Guidance Network via Robust Contextual Embedding for Efficient Point Cloud Registration}, 
  author={Wang, Jingtao and Liu, Xin and Dai, Luanyuan and Ma, Jiayi and Wei, Lifang and Yang, Changcai and Chen, Riqing},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  year={2023},
  volume={61},
  number={},
  pages={1-12},
  doi={10.1109/TGRS.2023.3266285}
}```

## Acknowledgement
This code is borrowed from [PointDSC](https://github.com/XuyangBai/PointDSC). If using the part of code related to data generation, testing and evaluation, please cite these papers:
```
@article{bai2021pointdsc,
  title={{PointDSC}: {R}obust {P}oint {C}loud {R}egistration using {D}eep {S}patial {C}onsistency},
  author={Xuyang Bai, Zixin Luo, Lei Zhou, Hongkai Chen, Lei Li, Zeyu Hu, Hongbo Fu and Chiew-Lan Tai},
  journal={CVPR},
  year={2021}
}
