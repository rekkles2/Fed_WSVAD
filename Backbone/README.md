# Video Feature Extraction Toolkit (Backbone)

A modular toolkit for extracting video features using the **Video Masked Autoencoder V2 (VideoMAE V2)** framework.
This backbone is part of the repository:

🔗 https://github.com/rekkles2/Fed_WSVAD

It is used in the paper:

> **"Dual-detector Re-optimization for Federated Weakly Supervised Video Anomaly Detection via Adaptive Dynamic Recursive Mapping"**  
> *(Accepted at IEEE Transactions on Industrial Informatics)*

---

## 📑 Table of Contents

1. [Introduction](#introduction)  
2. [Pretrained Model](#pretrained-model)  
3. [Workflow Overview](#workflow-overview)  
   - [RGB Frame Extraction](#rgb-frame-extraction)  
   - [Feature Directory Setup](#feature-directory-setup)  
4. [Configuration Guide](#configuration-guide)  
   - [main.py](#mainpy)  
   - [feature_extract.py](#feature_extractpy)  
   - [write_data_label_txt_new.py](#write_data_label_txt_newpy)  
   - [dataset_creater.py](#dataset_createrpy)  
5. [Feature Alignment](#feature-alignment)  
6. [Citation](#citation)  
7. [Related Resource](#related-resource)  

---

## 🚀 Introduction

This repository provides scripts for extracting video features using the **Video-MAE V2** model. It is designed to support feature preprocessing in weakly supervised video anomaly detection tasks.

---

## 🎯 Pretrained Model

You can download the pretrained Video-MAE V2 checkpoint from the link below:

👉 [Download Pretrained Model](https://drive.google.com/file/d/1xr1yeA2cxck4NCLX1qjAi3JU9qhRpfGr/view?usp=drive_link)

---

## 🛠 Workflow Overview

### 1. RGB Frame Extraction

Convert videos to RGB frames:
```bash
python RGB_extraction.py
```

**Parameters:**
- `root_folder`: Path to source videos  
- `output_root`: Path to save RGB frames

```python
root_folder = "F:/Backbone/video/shanghaitech"
output_root = "F:/Backbone/video/Rgb_Fig"
```

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

### 2. Feature Directory Setup

Prepare feature directories by running:
```bash
python feature_folder_generate.py
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

## ⚙️ Configuration Guide

Adjust paths and checkpoints in the following scripts:

### 🔹 main.py
```python
video_list     = "./shanghaitech.txt"
frame_folder   = "./video/Rgb_Fig/{video}"
delet_folder   = "./dataset/shanghaitech/features"
ft_folder      = "./dataset/shanghaitech/features_video/i3d/combine/shanghaitech"
feature_folder = "./shanghaitech/features_video/i3d/combine/{video}"
```

### 🔹 feature_extract.py
```python
data_path = "F:/Backbone"
ckpt_path = "path/to/checkpoint"
```

### 🔹 write_data_label_txt_new.py
```python
data_root = "F:/Backbone"
```

### 🔹 dataset_creater.py
```python
folder_path = "F:/Backbone/dataset/shanghaitech/features/i3d/rgb/shanghaitech"
output_file = "F:/Backbone/dataset/shanghaitech/features_video/i3d/combine/shanghaitech/feature.npy"
```

---

## 🧩 Feature Alignment

Align features with ground truth labels:
```bash
python feature_alignment.py
```

**Parameters:**
```python
feature_dir = "/VAD/shanghaitech/features_video/i3d/combine"
gt_path     = "/VAD/shanghaitech/GT/frame_label.pickle"
```

✅ All aligned feature files will be saved in the `shanghaitech` directory.

---

## 📚 Citation

If you use VideoMAE V2, please cite:
```bibtex
@inproceedings{wang2023videomae,
  title={Videomae v2: Scaling video masked autoencoders with dual masking},
  author={Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14549--14560},
  year={2023}
}
```


