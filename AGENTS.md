# AGENTS.md - UROP UNet + Kalman Filter Project

## Project Overview
UAV object tracking: UNet segmentation + Kalman Filter temporal smoothing.

## Key Commands
- **학습**: `python trainer.py --config config.yaml --epochs 50`
- **테스트 (필터 비교)**: `python tester.py --checkpoint checkpoints/demo_unet.pth --filter all --save-vis`
- **Q sweep**: `python tester.py --checkpoint checkpoints/demo_unet.pth --sweep`
- **합성 데이터 평가**: `python eval/run_phase3.py`

## Architecture
- **UNet**: VanillaUNet (480×480 → binary mask)
- **Filters**: Linear KF (CV), Constant Acceleration (CA), EKF (CTRV)
- **Pipeline**: Frame → UNet → Mask → Centroid → Kalman Predict/Update → Smoothed Position

## Key Files
- `model/Vanilla_UNet.py`: UNet architecture
- `filters/linear_kalman_filter.py`: Linear KF (CV model)
- `filters/constant_acceleration_filter.py`: CA filter
- `filters/extended_kalman_filter.py`: EKF (CTRV model)
- `combined_model/unet_kalman_combined.py`: Integration class
- `dataset/uav_dataset.py`: ANTI-UAV dataset loader
- `trainer.py`: Training with progress visualization
- `tester.py`: Testing with filter comparison & Q sweep
- `eval/metrics.py`: Evaluation metrics (IoU, Dice, CLE, Jitter)
- `config.yaml`: Main config
- `config_light.yaml`: Lightweight UNet config

## Conventions
- Batch size 1 recommended for Kalman (stateful)
- Reset Kalman between sequences (jump > 100px detection)
- UNet on GPU, Kalman on CPU
- Q parameter controls accuracy-stability tradeoff
