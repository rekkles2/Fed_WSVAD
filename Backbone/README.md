# Video Feature Extraction (Backbone)

A toolkit for extracting video features using the **Video Masked Autoencoder (Video-MAE V2)** framework.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [Pretrained Model](#pretrained-model)
4. [Workflow Overview](#workflow-overview)

   * [RGB Frame Extraction](#rgb-frame-extraction)
   * [Feature Directory Setup](#feature-directory-setup)
5. [Configuration Guide](#configuration-guide)

   * [main.py](#mainpy)
   * [feature\_extract.py](#feature_extractpy)
   * [write\_data\_label\_txt\_new.py](#write_data_label_txt_newpy)
   * [dataset\_creater.py](#dataset_createrpy)
6. [Feature Alignment](#feature-alignment)

---

## Introduction

This toolkit provides scripts to extract video features using the **Video Masked Autoencoder (Video-MAE V2)** model.

---

## Citation

Include the following BibTeX entry in your references:

```bibtex
@inproceedings{wang2023videomae,
  title={Videomae v2: Scaling video masked autoencoders with dual masking},
  author={Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14549--14560},
  year={2023}
}
```

---

## Pretrained Model

Download the pretrained Video-MAE V2 checkpoint:

[Download Pretrained Model](https://drive.google.com/file/d/1xr1yeA2cxck4NCLX1qjAi3JU9qhRpfGr/view?usp=drive_link)

---

## Workflow Overview

### RGB Frame Extraction

Run the frame extraction script to convert videos into RGB images:

```bash
python RGB_extraction.py
```

**Key Parameters:**

* `root_folder`: Path to the directory of source videos
* `output_root`: Path to save the extracted RGB frames

```python
# Example parameters
root_folder = "F:/Backbone/video/shanghaitech"
output_root = "F:/Backbone/video/Rgb_Fig"
```

**Sample Output Structure:**

```
Rgb_Fig/
├── 01_0014/
│   ├── img_00000.jpg
│   ├── img_00001.jpg
│   └── ...
└── 02_0035/
    ├── img_00000.jpg
    ├── img_00001.jpg
    └── ...
```

---

### Feature Directory Setup

Generate the directory structure for features by running:

```bash
python feature_folder_generate.py
```

**Generated Directory Structure:**

```
shanghaitech/
└── features_video/
    └── i3d/
        └── combine/
            ├── 01_0014/feature.npy
            ├── 02_0035/feature.npy
            └── ...
```

---

## Configuration Guide

Adjust the script parameters below to match your data paths and checkpoint locations.

### main.py

```python
video_list     = "./shanghaitech.txt"          # Path to the video list (train and test)
frame_folder   = "./video/Rgb_Fig/{video}"     # Path template for RGB frames
delet_folder   = "./dataset/shanghaitech/features"  # Directory for cleanup
ft_folder      = "./dataset/shanghaitech/features_video/i3d/combine/shanghaitech"
feature_folder = "./shanghaitech/features_video/i3d/combine/{video}"
```

### feature\_extract.py

```python
data_path = "F:/Backbone"    # Root directory for data
ckpt_path = "path/to/checkpoint"  # Pretrained model checkpoint path
```

### write\_data\_label\_txt\_new\.py

```python
data_root = "F:/Backbone"  # Root data directory for labels
```

### dataset\_creater.py

```python
folder_path = "F:/Backbone/dataset/shanghaitech/features/i3d/rgb/shanghaitech"  # Input feature directory
output_file = "F:/Backbone/dataset/shanghaitech/features_video/i3d/combine/shanghaitech/feature.npy"  # Merged feature file
```

---

## Feature Alignment

Align extracted features with ground truth labels by running:

```bash
python feature_alignment.py
```

**Alignment Parameters:**

* `feature_dir`: Directory containing combined feature files
* `gt_path`: Path to the ground truth frame label pickle file

```python
feature_dir = "/VAD/shanghaitech/features_video/i3d/combine"
gt_path     = "/VAD/shanghaitech/GT/frame_label.pickle"
```

## All generated feature files will be stored under the `shanghaitech` directory after alignment.
