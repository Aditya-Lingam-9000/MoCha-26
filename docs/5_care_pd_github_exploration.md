# CARE-PD GitHub Repository Exploration

This document details the analysis of the `TaatiTeam/CARE-PD` GitHub repository, which contains the official codebase used to train and evaluate models for the NeurIPS 2025 CARE-PD paper.

## 1. Core Architecture and Focus
Unlike the MoCha baseline (which is a lightweight CodaBench wrapper), this repository is a massive, highly parameterized Deep Learning framework designed for **Representation Learning** and **Multi-Dataset Transfer Learning** on High-End GPUs. 

It implements several state-of-the-art Sequence-to-Sequence Vision Transformers and Autoencoders.

## 2. Directory & File Analysis

### `model/` (The Backbones)
- **`backbone_loader.py`**: A central hub that instantiates 3D human pose models:
  - `MotionBERT` (DSTformer)
  - `MixSTE` (Cross-Attention Transformer)
  - `MotionAGFormer` (Graph-based Transformer)
  - `PoseFormerV2`
  - `MoMask` (Vector Quantized Variational Autoencoder - VQ-VAE)
  - `POTR` (Pose Transformer)
  - `MotionCLIP`
- **Implication for Us**: These are extremely heavy transformer models. Training them from scratch on a Ryzen 3 with 8GB RAM is impossible. However, the repository provides pre-trained weights for them.

### `data/` (The Dataloaders)
- **`dataloaders.py` and `*_datareader.py`**: Scripts designed to load the 9 cohort datasets (e.g., `bmclab_datareader.py`, `dne_datareader.py`).
- **`preprocessing/`**: Contains scripts converting raw SMPL parameters into `h36m`, `HumanML3D`, and `6D_SMPL` formats. 
- **`augmentations.py`**: Implements sequence augmentations like jitter, masking, and flipping (used heavily when training Neural Networks to prevent overfitting).

### `scripts/` (Bash Execution)
- **`download_models.sh`**: Uses `gdown` to download pre-trained checkpoints (approx. 1GB+) for the transformer backbones.
- **`eval_within_dataset.sh`, `eval_cross_dataset.sh`, `eval_lodo.sh`, `eval_mida.sh`**: Scripts mapping to the NeurIPS paper's 4 experimental setups:
  1. LOSO (Leave-One-Subject-Out)
  2. Cross-Dataset
  3. LODO (Leave-One-Dataset-Out)
  4. MIDA (Multi-domain In-Dataset Adaptation)

### Root Execution Scripts
- **`train.py` & `run.py`**: The main execution loops utilizing PyTorch `DataLoader` and `nn.DataParallel`. They expect GPU availability (`cuda`).
- **`eval_only.py`**: Evaluates pre-trained models using predefined `configs/*.json`.

## 3. Hardware Conflict & Resolution
The CARE-PD GitHub repo assumes you have a CUDA-enabled GPU and enough RAM to load large sequences into PyTorch DataLoaders. 

**Our Hardware:** Ryzen 3 Dual-Core, 8GB RAM, SSD, No GPU.

**Resolution:** 
We cannot use `train.py` or `run.py` from this repository. We cannot train `MotionBERT` or `MixSTE`. 

However, we **can** download the pre-trained weights (`scripts/download_models.sh`), instantiate one of the backbones on the CPU via `backbone_loader.py` in `eval()` mode, and use it exactly like we planned to use the baseline `EvaluatorModelWrapper`—as an **offline feature extractor**. 

If the baseline BiGRU feature extractor fails to reach `0.70` Macro F1, we can switch to extracting features using the pre-trained `MoMask` or `MotionCLIP` models provided in this repository, before feeding those rich embeddings into our CPU-friendly LightGBM model.
