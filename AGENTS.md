# AGENTS.md - UROP UNet + Kalman Filter Project

## Project Overview
UAV object tracking project integrating UNet segmentation with Kalman Filter temporal smoothing. See [README.md](README.md) for detailed architecture and usage.

## Key Commands
- **Import verification**: `python test_imports.py`
- **Quick test**: `python quick_test.py`
- **Integration test**: `python experiments/phase2_test.py`
- **Interactive testing**: `python phase2_interactive.py`

## Architecture
- **UNet**: Generates object masks from 480×480 frames
- **Kalman Filter**: 2D constant-velocity model for position smoothing
- **Integration**: Centroid extraction → Kalman predict/update → mask smoothing

## Conventions
- **Batch size**: Hardcoded to 1; batching logic incomplete
- **State management**: Reset Kalman between sequences
- **Device handling**: UNet on GPU, Kalman on CPU
- **Coordinates**: Watch for x/y vs row/col indexing

## Pitfalls
- Stateful Kalman carries state between calls; use `.reset()`
- Single-object assumption per frame
- No pre-trained weights; requires training
- Missing UNet_center.py implementation

## Key Files
- [combined_model/unet_kalman_combined.py](combined_model/unet_kalman_combined.py): Main integration class
- [filters/linear_kalman_filter.py](filters/linear_kalman_filter.py): Kalman implementation
- [model/Vanilla_UNet.py](model/Vanilla_UNet.py): UNet architecture
- [dataset/uav_dataset.py](dataset/uav_dataset.py): Data loading
- [config.yaml](config.yaml): Configuration parameters