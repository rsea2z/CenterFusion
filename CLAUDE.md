# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CenterFusion is a center-based radar and camera fusion approach for 3D object detection on the nuScenes dataset. It achieves NDS 0.449, mAP 0.326 on nuScenes Test using PyTorch Lightning as the training framework.

## Common Commands

### Setup & Installation
```bash
git clone --recursive https://github.com/michuanhaohao/CenterFusion.git
conda create -n centerfusion python=3.9
conda activate centerfusion
pip install torch torchvision
pip install -r requirements.txt
# Install system deps: libsm6, libxext6, ffmpeg
python src/convert_nuScenes.py
```

### Training
```bash
python src/main.py --cfg configs/CenterFusion_Middle.yaml
# Override config values inline
python src/main.py --cfg configs/centerfusion_debug.yaml TRAIN.LR=0.001 TRAIN.EPOCHS=100
```

### Evaluation
Set `EVAL: true` in config, then run:
```bash
python src/main.py --cfg configs/centerfusion_debug.yaml
```

### Demo / Visualization
```bash
python src/demo.py --cfg configs/centerfusion_debug.yaml --split val --save
python src/demo.py --cfg configs/centerfusion_debug.yaml --split val --sample <sample_token>
```

### Inference on Video/Images
```bash
python src/inference.py --cfg configs/centerfusion_debug.yaml --input ./video.mp4 --save
python src/inference.py --cfg configs/centerfusion_debug.yaml --input webcam --save
```

## Architecture

### Data Flow
```
Radar Point Cloud + Camera Image
    │
    ▼
Dataset (nuscenes.py) → Generates heatmaps, depth maps, frustum associations
    │
    ▼
DLA-34 Backbone (dla.py) → Extracts image features
    │
    ▼
Fusion Module (fusionModules.py) → Associates radar with image features via frustum
    │
    ▼
Detection Heads (detectHeads.py) → Outputs heatmap, depth, rotation, dimension, velocity
    │
    ▼
Decode (decode.py) → NMS, top-K extraction
    │
    ▼
Post-processing (postProcess.py, detector.py) → 3D box projection, coordinate transform
```

### Key Modules

| Module | Location | Role |
|--------|----------|------|
| `GenericDataset` | `src/lib/dataset/generic_dataset.py` | Base dataset: image loading, augmentation, point cloud processing |
| `nuScenes` | `src/lib/dataset/datasets/nuscenes.py` | nuScenes-specific dataset with multi-camera support, frustum association |
| `Detector` | `src/lib/detector.py` | Main inference pipeline: preprocess → forward → postprocess |
| `BaseModel` | `src/lib/model/networks/base_model.py` | Backbone + heads assembly |
| `DLASeg` | `src/lib/model/networks/dla.py` | DLA-34 backbone with deformable convolutions |
| `CenterFusionHead` | `src/lib/model/networks/detectHeads.py` | Fusion detection heads (depth2, rotation2 from radar) |
| `ConcateCombiner` | `src/lib/model/networks/fusionModules.py` | Radar-image feature fusion via concatenation |
| `ModelWithLoss` | `src/lib/model/modelWithLoss.py` | PyTorch Lightning LightningModule wrapping model + loss |
| `GenericLoss` | `src/lib/model/genericLoss.py` | Aggregates all loss terms |
| `Trainer` | `src/lib/trainer.py` | PyTorch Lightning Trainer config with DDP + WandbLogger |

### Fusion Strategies
Configured via `MODEL.FUSION_STRATEGY` in config:
- `middle` (default): Radar features fused at intermediate layers via frustum association
- `early`: Concatenate radar/image features earlier

### Detection Heads
- **2D**: heatmap, reg, widthHeight
- **3D**: depth, rotation, dimension, amodal_offset
- **nuScenes-specific**: velocity, nuscenes_att, depth2, rotation2
- **Radar-derived**: depth2, rotation2 (improved estimates from radar fusion)

## Configuration

Configs live in `configs/`. Key config files:
- `CenterFusion_Middle.yaml` — Production training (4 GPUs, 200 epochs, DLA pretrained)
- `centerfusion_debug.yaml` — Development (mini_val, single GPU)
- `CenterNet.yaml` — Baseline (no fusion)

Config is managed by **yacs** (`src/lib/config/default.py`). All config values are accessible as `cfg.SECTION.KEY`.

## Side Vehicle State Classification

Extended capability for classifying side vehicle states (moving, parking):
- `tools/train_side_state_model.py` — Train classifier on demo outputs
- `tools/label_states.py` — Label states from demo output JSON
- `tools/analyze_side_features.py` — Feature importance analysis
- Demo flag: `--state-classify --side-model PATH --side-gate DIST`

## Pre-trained Models

Download from README and place in `models/` directory. Expected naming: `centerfusion_e60`, `centerfusion_e230`, etc.
