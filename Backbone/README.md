# Video Feature Extraction Toolkit (Backbone)

A modular toolkit for extracting video features using the **Video Masked Autoencoder V2 (VideoMAE V2)** framework.

This backbone is part of the repository: [rekkles2/Fed_WSVAD](https://github.com/rekkles2/Fed_WSVAD)

It is used in the paper:

> **"Dual-detector Re-optimization for Federated Weakly Supervised Video Anomaly Detection via Adaptive Dynamic Recursive Mapping"**  
> *(Accepted at IEEE Transactions on Industrial Informatics)*

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Pretrained Model](#-pretrained-model)
- [Workflow Overview](#-workflow-overview)
  - [RGB Frame Extraction](#rgb-frame-extraction)
  - [Feature Directory Setup](#feature-directory-setup)
  - [Feature Extraction](#feature-extraction)
  - [Feature Alignment](#feature-alignment)
- [Citation](#-citation)
---

## 🚀 Introduction

This repository provides scripts for extracting video features using the **Video-MAE V2** model.  
It is designed to support feature preprocessing in weakly supervised video anomaly detection tasks.

---

## 🎯 Pretrained Model

Download the pretrained Video-MAE V2 checkpoint:

👉 [Download Pretrained Model](https://drive.google.com/file/d/1xr1yeA2cxck4NCLX1qjAi3JU9qhRpfGr/view?usp=drive_link)

---

## 🛠 Workflow Overview

### <a id="rgb-frame-extraction"></a>1. RGB Frame Extraction

Convert videos to RGB frames:

```bash
python RGB_extraction.py \
  --root_folder F:/Backbone/video/shanghaitech \
  --output_root F:/Backbone/video/Rgb_Fig
````

**Output Structure:**

```
Rgb_Fig/
├── 01_0014/
│   ├── img_00000.jpg
│   └── ...
└── 02_0035/
    ├── img_00000.jpg
    └── ...
```

---

### <a id="feature-directory-setup"></a>2. Feature Directory Setup

Generate feature directories:

```bash
python feature_folder_generate.py \
  --data_root shanghaitech/features_video/i3d/combine
```

**Generated Structure:**

```
shanghaitech/
└── features_video/
    └── i3d/
        └── combine/
            ├── 01_0014/feature.npy
            └── 02_0035/feature.npy
```

---

### <a id="feature-extraction"></a>3. Feature Extraction

Update paths in the scripts to match your file system, then run:

```bash
python main.py
```

Key configurations:

* **main.py**

  ```python
  video_list     = "./shanghaitech.txt"
  frame_folder   = "./video/Rgb_Fig/{video}"
  delet_folder   = "./dataset/shanghaitech/features"
  ft_folder      = "./dataset/shanghaitech/features_video/i3d/combine/shanghaitech"
  feature_folder = "./shanghaitech/features_video/i3d/combine/{video}"
  ```
* **feature\_extract.py**

  ```python
  data_path = "F:/Backbone"
  ckpt_path = "path/to/checkpoint"
  ```
* **write\_data\_label\_txt\_new\.py**

  ```python
  data_root = "F:/Backbone"
  ```
* **dataset\_creater.py**

  ```python
  folder_path = "F:/Backbone/dataset/shanghaitech/features/i3d/rgb/shanghaitech"
  output_file = "F:/Backbone/dataset/shanghaitech/features_video/i3d/combine/shanghaitech/feature.npy"
  ```

---

### <a id="feature-alignment"></a>4. Feature Alignment

Align features with ground truth labels:

```bash
python feature_alignment.py \
  --feature_dir /VAD/shanghaitech/features_video/i3d/combine \
  --gt_path /VAD/shanghaitech/GT/frame_label.pickle
```

Aligned feature files will be saved under the `shanghaitech` directory.

---

## 📚 Citation

If you use VideoMAE V2 as backbone, please cite:
```bibtex
@ARTICLE{11036561,
  author={Su, Yong and Li, Jiahang and An, Simin and Xu, Hengpeng and Peng, Weilong},
  journal={IEEE Transactions on Industrial Informatics},
  title={Dual-Detector Reoptimization for Federated Weakly Supervised Video Anomaly Detection via Adaptive Dynamic Recursive Mapping},
  year={2025},
  volume={},
  number={},
  pages={1-11},
  keywords={Adaptation models;Training;Anomaly detection;Feature extraction;Surveillance;Optimization;Accuracy;Privacy;Detectors;Semantics;Adaptive dynamic recursive mapping;adaptive local aggregation;federated;scene-similarity;video anomaly detection (VAD);weakly supervised},
  doi={10.1109/TII.2025.3574406}
}
```

```bibtex
@inproceedings{wang2023videomae,
  title={Videomae v2: Scaling video masked autoencoders with dual masking},
  author={Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14549--14560},
  year={2023}
}
```

