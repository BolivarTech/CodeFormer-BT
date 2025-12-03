<p align="center">
  <img src="assets/CodeFormer_logo.png" height="110" alt="CodeFormer Logo">
</p>

<h1 align="center">CodeFormer-BT</h1>

<p align="center">
  <strong>Robust Blind Face Restoration with Codebook Lookup Transformer</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2206.11253"><img src="https://img.shields.io/badge/arXiv-2206.11253-b31b1b.svg" alt="arXiv"></a>
  <a href="https://shangchenzhou.com/projects/CodeFormer/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-NTU%20S--Lab%201.0-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10--3.13-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/sczhou/CodeFormer"><img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue" alt="Hugging Face"></a>
  <a href="https://replicate.com/sczhou/codeformer"><img src="https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue" alt="Replicate"></a>
  <a href="https://openxlab.org.cn/apps/detail/ShangchenZhou/CodeFormer"><img src="https://img.shields.io/badge/Demo-%F0%9F%90%BC%20OpenXLab-blue" alt="OpenXLab"></a>
</p>

---

> **Note:** This is a fork of the original [CodeFormer](https://github.com/sczhou/CodeFormer) project, updated for Python 3.13 and modern CUDA versions. For the original documentation, see [README-ORG.md](README-ORG.md).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Face Restoration](#face-restoration)
  - [Video Enhancement](#video-enhancement)
  - [GPU-Accelerated Processing](#gpu-accelerated-video-processing-nvdecnvenc)
  - [Face Colorization](#face-colorization)
  - [Face Inpainting](#face-inpainting)
- [Training](#training)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

CodeFormer is a state-of-the-art blind face restoration model that leverages a **Codebook Lookup Transformer** to restore degraded face images. Published at **NeurIPS 2022**, it achieves remarkable results in face restoration, colorization, and inpainting tasks.

<p align="center">
  <img src="assets/network.jpg" width="800" alt="CodeFormer Architecture">
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| **Face Restoration** | Restore degraded, blurry, or low-quality face images |
| **Face Colorization** | Colorize black & white or faded photographs |
| **Face Inpainting** | Fill in missing or masked regions of face images |
| **Video Enhancement** | Process video files with face restoration |
| **GPU Image Processing** | NVIDIA CUDA acceleration for neural network inference |
| **GPU Video Encoding** | NVIDIA NVENC/NVDEC hardware acceleration for video |
| **Background Enhancement** | Optional Real-ESRGAN integration for full image enhancement |

---

## Requirements

| Component | Version |
|-----------|---------|
| **Python** | 3.10 - 3.13 |
| **PyTorch** | >= 2.6.0 |
| **CUDA** | 12.4 - 13.0 |
| **GPU VRAM** | 4 GB minimum |

---

## Installation

### 1. Clone Repository

```bash
# Clone this fork (Python 3.13 compatible)
git clone https://github.com/BolivarTech/CodeFormer-BT.git
cd CodeFormer-BT

# Or clone the original repository
# git clone https://github.com/sczhou/CodeFormer.git
# cd CodeFormer
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### 3. Install PyTorch

Select according to your CUDA version:

```bash
# CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 13.0 (experimental)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install BasicSR

```bash
pip install -e .
```

### 6. Download Pretrained Models

```bash
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer
```

### 7. (Optional) Install dlib

dlib provides better face identity preservation. Choose one option:

<details>
<summary><strong>Option A: pip install (requires C++ compiler)</strong></summary>

```bash
pip install dlib
```

**Prerequisites (Windows):**
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"

</details>

<details>
<summary><strong>Option B: Precompiled wheel (Windows - no compilation)</strong></summary>

Download the appropriate wheel from [Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x):

```bash
# Example for Python 3.13
pip install dlib-19.24.99-cp313-cp313-win_amd64.whl
```

</details>

<details>
<summary><strong>Option C: Build from source</strong></summary>

```bash
pip install cmake
pip install dlib --verbose
```

</details>

After installing dlib, download its models:

```bash
python scripts/download_pretrained_models.py dlib
```

> **Note:** dlib is optional. By default, CodeFormer uses RetinaFace for face detection.

### 8. (Optional) Install FFmpeg for Video Processing

FFmpeg is required for video enhancement features.

<details>
<summary><strong>Windows (Chocolatey - recommended)</strong></summary>

[Chocolatey](https://chocolatey.org/install) is a package manager for Windows.

```powershell
# Install Chocolatey (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg with NVENC/NVDEC support
choco install ffmpeg-full -y

# Verify installation
ffmpeg -version
```

Other useful packages:
```powershell
choco install git python313 cuda -y
```

</details>

<details>
<summary><strong>Windows (Manual)</strong></summary>

1. Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (full build for NVENC/NVDEC support)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable

```bash
# Verify installation
ffmpeg -version
```

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# With NVENC support (requires NVIDIA drivers)
sudo apt install ffmpeg nvidia-cuda-toolkit
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install ffmpeg
```

</details>

> **Note:** For GPU-accelerated video encoding/decoding, use FFmpeg builds with NVENC/NVDEC support.

---

## Quick Start

```bash
# Basic face restoration
python inference_codeformer.py -w 0.7 --input_path inputs/whole_imgs

# With background enhancement
python inference_codeformer.py -w 0.7 --bg_upsampler realesrgan --face_upsample --input_path inputs/whole_imgs
```

When running, you'll see a banner showing the processing device:

```
============================================================
                      IMAGE PROCESSING
============================================================
  Device:    GPU (NVIDIA CUDA)
  PyTorch:   2.6.0+cu124
  GPU:       Quadro P1000
  VRAM:      4.0 GB
  CUDA:      12.4
============================================================
```

---

## Usage

### Fidelity Weight (`-w`)

| Weight | Effect |
|--------|--------|
| `0.0 - 0.4` | Higher quality, may alter identity |
| `0.5 - 0.6` | Balanced (recommended) |
| `0.7 - 1.0` | Better identity preservation |

### Background Upsampler (`--bg_upsampler`)

| Option | Scale | Model | Description |
|--------|-------|-------|-------------|
| `realesrgan` | 2x | RealESRGAN_x2plus | Standard quality, faster |
| `realesrgan_x4` | 4x | RealESRGAN_x4plus | Higher quality, larger output |

Models are downloaded automatically on first use.

### Face Restoration

```bash
# Aligned faces (512x512)
python inference_codeformer.py -w 0.5 --has_aligned --input_path inputs/cropped_faces

# Whole image
python inference_codeformer.py -w 0.7 --input_path inputs/whole_imgs

# With background enhancement (2x)
python inference_codeformer.py -w 0.7 --bg_upsampler realesrgan --face_upsample --input_path inputs/whole_imgs

# With background enhancement (4x - higher quality)
python inference_codeformer.py -w 0.7 --bg_upsampler realesrgan_x4 --face_upsample --input_path inputs/whole_imgs
```

### Video Enhancement

```bash
python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path inputs/video.mp4
```

#### First and Last Frame Extraction

Extract and restore only the first and last frames of a video (useful for quick previews or thumbnail generation):

```bash
# Basic extraction
python inference_codeformer.py -w 0.7 --first_last --input_path inputs/video.mp4

# With background and face upsampling (2x)
python inference_codeformer.py -w 0.7 --first_last --bg_upsampler realesrgan --face_upsample --input_path inputs/video.mp4

# With background and face upsampling (4x - higher quality)
python inference_codeformer.py -w 0.7 --first_last --bg_upsampler realesrgan_x4 --face_upsample --input_path inputs/video.mp4
```

**Output:** `frame_first.png` and `frame_last.png` in `results/<video_name>_first_last/final_results/`

#### GPU-Accelerated Video Processing (NVDEC/NVENC)

Video processing automatically detects and uses NVIDIA GPU for both **decoding** and **encoding** when available:

| Operation | GPU Mode | CPU Mode | Speed Improvement |
|-----------|----------|----------|-------------------|
| **Decoding** | NVDEC/CUVID | Software | ~2-3x faster |
| **Encoding** | NVENC | libx264 | ~5-10x faster |

**Supported CUDA Decoders:**
- H.264, HEVC/H.265, VP8, VP9, AV1, MPEG-1/2/4, MJPEG, VC1

When processing videos, you'll see clear banners indicating the mode:

<details>
<summary><strong>Decoding Banners</strong></summary>

**GPU Decoding (Green):**
```
============================================================
                       VIDEO DECODING
============================================================
  Mode:      GPU (NVIDIA NVDEC/CUVID)
  FFmpeg:    8.0.1
  NVDEC:     av1_cuvid, h264_cuvid, hevc_cuvid, mjpeg_cuvid
============================================================
```

**CPU Decoding (Yellow):**
```
============================================================
                       VIDEO DECODING
============================================================
  Mode:      CPU (Software Decoder)
  FFmpeg:    8.0.1
============================================================
```
</details>

<details>
<summary><strong>Encoding Banners</strong></summary>

**GPU Encoding (Green):**
```
============================================================
                       VIDEO ENCODING
============================================================
  Mode:      GPU (NVIDIA NVENC)
  Codec:     h264_nvenc
  FFmpeg:    8.0.1
  NVENC:     av1_nvenc, h264_nvenc, hevc_nvenc
============================================================
```

**CPU Encoding (Yellow):**
```
============================================================
                       VIDEO ENCODING
============================================================
  Mode:      CPU (libx264)
  Codec:     libx264
  FFmpeg:    8.0.1
============================================================
```
</details>

<details>
<summary><strong>CUDA Not Available Warning (Red)</strong></summary>

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          WARNING: CUDA DECODING NOT AVAILABLE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  NVDEC decoders: Not found
  CUDA hwaccel:   No

  Falling back to CPU decoding (slower)

  To enable CUDA decoding:
  1. Install NVIDIA GPU drivers
  2. Use FFmpeg with NVENC/NVDEC support
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
</details>

> **Note:** Download FFmpeg with NVENC/NVDEC support from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (full build).

### Face Colorization

```bash
python inference_colorization.py --input_path inputs/gray_faces
```

### Face Inpainting

```bash
python inference_inpainting.py --input_path inputs/masked_faces
```

---

## Training

Three-stage training pipeline:

| Stage | Description | Config |
|-------|-------------|--------|
| **I** | VQGAN codebook | `VQGAN_512_ds32_nearest_stage1.yml` |
| **II** | Transformer (w=0) | `CodeFormer_stage2.yml` |
| **III** | Controllable (w=1) | `CodeFormer_stage3.yml` |

```bash
# Stage I
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 \
    basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher pytorch

# Stage II
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4322 \
    basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch

# Stage III
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4323 \
    basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher pytorch
```

See [Training Documentation](docs/train.md) for details.

---

## Project Structure

```
CodeFormer/
├── basicsr/                 # Training framework
│   ├── archs/              # Network architectures
│   ├── models/             # Training models
│   └── data/               # Dataset loaders
├── facelib/                # Face processing utilities
│   ├── detection/          # Face detectors
│   ├── parsing/            # Face parsing
│   └── utils/              # Helpers
├── inference_*.py          # Inference scripts
├── scripts/                # Utility scripts
├── options/                # Training configs
└── weights/                # Pretrained models
```

---

## Citation

```bibtex
@inproceedings{zhou2022codeformer,
    author = {Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
    title = {Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
    booktitle = {NeurIPS},
    year = {2022}
}
```

---

## License

Licensed under [NTU S-Lab License 1.0](LICENSE).

---

<p align="center">
  <strong>Original Authors:</strong> <a href="https://shangchenzhou.com/">Shangchen Zhou</a>, <a href="https://ckkelvinchan.github.io/">Kelvin C.K. Chan</a>, <a href="https://li-chongyi.github.io/">Chongyi Li</a>, <a href="https://www.mmlab-ntu.com/person/ccloy/">Chen Change Loy</a>
  <br>
  S-Lab, Nanyang Technological University
</p>
