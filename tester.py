"""
Phase 3: 실제 테스트 데이터 평가

학습된 UNet 체크포인트를 로드하여 test 시퀀스에서 평가.
Baseline (UNet only) vs UNet + Kalman Filter 비교.

사용법:
    python tester.py --checkpoint checkpoints/demo_unet.pth
    python tester.py --checkpoint checkpoints/demo_unet.pth --save-vis
"""

import argparse
import json
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.Vanilla_UNet import VanillaUNet
from dataset.uav_dataset import UAVTrackingDataset
from filters.linear_kalman_filter import KalmanFilter
from eval.metrics import SequenceMetrics, compute_iou, compute_dice


class TestPairDataset(torch.utils.data.Dataset):
    """테스트용 Dataset wrapper — (image, mask, centroid) 반환"""
    def __init__(self, base_dataset, image_size=(480, 480)):
        self.base = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, targets = self.base[idx]
        mask = targets['mask'].float().unsqueeze(0) / 255.0
        centroids = targets['centroids']  # (N, 2) — 원본 해상도 기준

        # 원본 마스크 크기 (centroid 스케일링용)
        orig_h, orig_w = mask.shape[-2], mask.shape[-1]
        target_h, target_w = self.image_size

        # 이미지가 텐서면 마스크 크기 맞추기
        if isinstance(image, torch.Tensor):
            c, h, w = image.shape
            if (h, w) != mask.shape[-2:]:
                mask = TF.resize(mask, (h, w), interpolation=TF.InterpolationMode.NEAREST)
                target_h, target_w = h, w

        # 첫 번째 객체의 중심점 (단일 객체 가정)
        if centroids.shape[0] > 0:
            center = centroids[0].clone()  # [x, y]
            # 원본 해상도 → 타겟 해상도로 스케일링
            center[0] = center[0] * target_w / orig_w  # x
            center[1] = center[1] * target_h / orig_h  # y
        else:
            center = torch.tensor([0.0, 0.0])

        return image, mask, center


def load_model(checkpoint_path, model_cfg, device):
    """체크포인트에서 UNet 모델 로드"""
    unet_config = {
        'in_channels': model_cfg.get('in_channels', 3),
        'start_out_channels': model_cfg.get('start_out_channels', 32),
        'num_class': model_cfg.get('num_class', 1),
        'size': model_cfg.get('size', 4),
        'padding': model_cfg.get('padding', 1),
    }

    model = VanillaUNet(**unet_config)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f'✓ 모델 로드 완료: {checkpoint_path}')
    if 'epoch' in ckpt:
        print(f'  학습 epoch: {ckpt["epoch"]}, val_loss: {ckpt.get("val_loss", "N/A")}')

    return model, unet_config


def extract_center_from_mask(mask_np, threshold=0.5):
    """마스크에서 중심점 추출"""
    binary = (mask_np > threshold).astype(np.uint8)
    if np.sum(binary) == 0:
        return None
    indices = np.argwhere(binary > 0)
    center_y, center_x = np.mean(indices, axis=0)
    return np.array([center_x, center_y], dtype=np.float32)


def test_baseline(model, test_loader, device):
    """Baseline: UNet만 사용 (Kalman 없이)"""
    metrics = SequenceMetrics()

    with torch.no_grad():
        for images, masks_gt, centers_gt in tqdm(test_loader, desc='[Baseline] UNet Only'):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for b in range(images.size(0)):
                pred_mask = probs[b, 0].cpu().numpy()
                gt_mask = masks_gt[b, 0].numpy()

                # GT 중심점: resize된 마스크에서 직접 계산 (좌표 불일치 방지)
                gt_center = extract_center_from_mask(gt_mask)
                # 예측 중심점
                pred_center = extract_center_from_mask(pred_mask)

                metrics.update(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    raw_center=pred_center,
                    gt_center=gt_center,
                    kalman_center=pred_center,  # baseline은 필터 없음
                )

    return metrics.summarize()


def test_with_kalman(model, test_loader, kalman_cfg, device):
    """UNet + Kalman Filter (시퀀스 전환 감지 시 리셋)"""
    metrics = SequenceMetrics()

    Q_scale = kalman_cfg.get('process_noise', 1.0)
    R_scale = kalman_cfg.get('measurement_noise', 0.5)
    dt = kalman_cfg.get('dt', 1.0)

    # 시퀀스 전환 감지 임계값 (pixels)
    # 연속 프레임에서 중심점이 이만큼 이상 점프하면 새 시퀀스로 판단
    JUMP_THRESHOLD = 100.0

    kf = None
    prev_center = None

    with torch.no_grad():
        for images, masks_gt, centers_gt in tqdm(test_loader, desc='[Proposed] UNet + Kalman'):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for b in range(images.size(0)):
                pred_mask = probs[b, 0].cpu().numpy()
                gt_mask = masks_gt[b, 0].numpy()

                # GT 중심점: resize된 마스크에서 직접 계산
                gt_center = extract_center_from_mask(gt_mask)

                # UNet에서 중심점 추출
                raw_center = extract_center_from_mask(pred_mask)

                # 시퀀스 전환 감지 → Kalman 리셋
                need_reset = False
                if kf is None:
                    need_reset = True
                elif raw_center is not None and prev_center is not None:
                    jump = np.linalg.norm(raw_center - prev_center)
                    if jump > JUMP_THRESHOLD:
                        need_reset = True

                if need_reset:
                    init_pos = raw_center if raw_center is not None else np.array([240.0, 240.0])
                    x0 = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float32)
                    kf = KalmanFilter(
                        dt=dt,
                        x0=x0,
                        Q=np.eye(4, dtype=np.float32) * Q_scale,
                        R=np.eye(2, dtype=np.float32) * R_scale,
                        P=np.eye(4, dtype=np.float32) * 100.0,
                    )

                # Kalman Filter 적용
                kf.predict()
                if raw_center is not None:
                    kf.update(raw_center)
                kalman_center = kf.get_position()

                prev_center = raw_center if raw_center is not None else prev_center

                metrics.update(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    raw_center=raw_center,
                    gt_center=gt_center,
                    kalman_center=kalman_center,
                )

    return metrics.summarize()


def plot_test_results(baseline_res, kalman_res, save_path):
    """테스트 결과 비교 바 차트"""
    metrics_to_plot = {
        'mIoU': (baseline_res['mIoU'], kalman_res['mIoU']),
        'Dice': (baseline_res['mean_dice'], kalman_res['mean_dice']),
        'CLE\n(lower=better)': (baseline_res['mean_CLE_raw'], kalman_res['mean_CLE_kalman']),
        'Jitter\n(lower=better)': (baseline_res['jitter_raw'], kalman_res['jitter_kalman']),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for idx, (name, (base_val, kal_val)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]
        bars = ax.bar(['UNet Only', 'UNet+Kalman'], [base_val, kal_val],
                     color=['#e74c3c', '#3498db'], alpha=0.8)
        ax.set_title(name, fontsize=12)
        ax.set_ylabel('Score')

        # 값 표시
        for bar, val in zip(bars, [base_val, kal_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Phase 3: Test Results — Baseline vs UNet+Kalman', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ 결과 차트 저장: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Phase 3: 실제 테스트 데이터 평가')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/demo_unet.pth')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='테스트 샘플 수 제한')
    parser.add_argument('--save-dir', default='eval/test_results')
    parser.add_argument('--save-vis', action='store_true',
                        help='시각화 결과 저장')
    args = parser.parse_args()

    # Config 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    kalman_cfg = cfg.get('kalman', {})

    device = torch.device(args.device)
    print(f'Device: {device}')

    # === 테스트 데이터셋 로드 ===
    root = dataset_cfg.get('root')
    root_mask = dataset_cfg.get('root_mask')
    image_size = tuple(dataset_cfg.get('image_size', [480, 480]))
    test_sequences = dataset_cfg.get('test_sequences', 1)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # test 시퀀스만 로드 (train과 겹치지 않게 offset 사용)
    # train_sequences + val_sequences 이후의 시퀀스를 test로 사용
    train_seq = dataset_cfg.get('train_sequences', 4)
    val_seq = dataset_cfg.get('val_sequences', 1)
    total_for_test = train_seq + val_seq + test_sequences

    print(f'\n데이터셋 로드 중...')
    print(f'  Root: {root}')
    print(f'  Test sequences: {test_sequences} (offset: {train_seq + val_seq})')

    # 전체 데이터 로드 후 test 부분만 추출
    full_dataset = UAVTrackingDataset(
        root, root_mask,
        transforms=transform,
        num_sequences=total_for_test
    )

    # train+val 이후의 데이터를 test로 사용
    # train_sequences * frames_per_seq 만큼 offset
    total_samples = len(full_dataset)
    test_start = int(total_samples * (train_seq + val_seq) / total_for_test)

    if args.max_samples is not None:
        test_end = min(test_start + args.max_samples, total_samples)
    else:
        test_end = total_samples

    test_indices = list(range(test_start, test_end))
    test_subset = Subset(full_dataset, test_indices)
    test_ds = TestPairDataset(test_subset, image_size=image_size)

    print(f'  Test 샘플 수: {len(test_ds)}')

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.get('num_workers', 0),
    )

    # === 모델 로드 ===
    model, unet_config = load_model(args.checkpoint, model_cfg, device)

    # === 평가 실행 ===
    print(f'\n{"="*60}')
    print('Phase 3: 실제 테스트 데이터 평가')
    print(f'{"="*60}')

    # 1. Baseline (UNet only)
    print('\n[1/2] Baseline 평가 (UNet only)...')
    baseline_result = test_baseline(model, test_loader, device)

    # 2. UNet + Kalman
    print('\n[2/2] UNet + Kalman Filter 평가...')
    kalman_result = test_with_kalman(model, test_loader, kalman_cfg, device)

    # === 결과 출력 ===
    print(f'\n{"="*70}')
    print(f'{"지표":<25} {"UNet Only":<15} {"UNet+Kalman":<15} {"차이":<15}')
    print(f'{"-"*70}')
    print(f'{"mIoU":<25} {baseline_result["mIoU"]:<15.4f} {kalman_result["mIoU"]:<15.4f} '
          f'{kalman_result["mIoU"] - baseline_result["mIoU"]:+.4f}')
    print(f'{"Dice":<25} {baseline_result["mean_dice"]:<15.4f} {kalman_result["mean_dice"]:<15.4f} '
          f'{kalman_result["mean_dice"] - baseline_result["mean_dice"]:+.4f}')
    print(f'{"CLE (pixels)":<25} {baseline_result["mean_CLE_raw"]:<15.4f} {kalman_result["mean_CLE_kalman"]:<15.4f} '
          f'{baseline_result["mean_CLE_raw"] - kalman_result["mean_CLE_kalman"]:+.4f}')
    print(f'{"Jitter (px/frame²)":<25} {baseline_result["jitter_raw"]:<15.4f} {kalman_result["jitter_kalman"]:<15.4f} '
          f'{kalman_result["jitter_reduction"]:+.1f}%')
    print(f'{"Detection Rate":<25} {baseline_result["detection_rate_raw"]:<15.4f} {kalman_result["detection_rate_kalman"]:<15.4f}')
    print(f'{"Smoothness Ratio":<25} {"---":<15} {kalman_result["smoothness_ratio"]:<15.4f} {"(>1=개선)"}')
    print(f'{"-"*70}')

    # === 결과 저장 ===
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'checkpoint': args.checkpoint,
        'test_samples': len(test_ds),
        'kalman_config': {
            'Q': kalman_cfg.get('process_noise'),
            'R': kalman_cfg.get('measurement_noise'),
            'dt': kalman_cfg.get('dt'),
        },
        'baseline': baseline_result,
        'unet_kalman': kalman_result,
    }

    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n✓ 결과 저장: {results_path}')

    # 시각화
    if args.save_vis:
        plot_path = save_dir / 'test_comparison.png'
        plot_test_results(baseline_result, kalman_result, plot_path)

    # 최종 요약
    print(f'\n{"="*60}')
    cle_improvement = baseline_result['mean_CLE_raw'] - kalman_result['mean_CLE_kalman']
    if cle_improvement > 0:
        print(f'✅ Kalman Filter가 위치 추적을 {cle_improvement:.2f}px 개선')
    else:
        print(f'⚠️ 현재 설정에서 Kalman Filter 효과 미미 (Q/R 튜닝 필요)')

    print(f'   Jitter 감소: {kalman_result["jitter_reduction"]:.1f}%')
    print(f'   Smoothness Ratio: {kalman_result["smoothness_ratio"]:.2f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
