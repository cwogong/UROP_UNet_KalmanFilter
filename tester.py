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
from filters.extended_kalman_filter import ExtendedKalmanFilter
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


def test_with_ekf(model, test_loader, kalman_cfg, device):
    """UNet + Extended Kalman Filter (CTRV 모션 모델)"""
    metrics = SequenceMetrics()

    Q_scale = kalman_cfg.get('process_noise', 0.1)
    R_scale = kalman_cfg.get('measurement_noise', 0.5)
    dt = kalman_cfg.get('dt', 1.0)

    JUMP_THRESHOLD = 100.0

    ekf = None
    prev_center = None

    with torch.no_grad():
        for images, masks_gt, centers_gt in tqdm(test_loader, desc='[Proposed] UNet + EKF'):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for b in range(images.size(0)):
                pred_mask = probs[b, 0].cpu().numpy()
                gt_mask = masks_gt[b, 0].numpy()
                gt_center = extract_center_from_mask(gt_mask)
                raw_center = extract_center_from_mask(pred_mask)

                # 시퀀스 전환 감지 → EKF 리셋
                need_reset = False
                if ekf is None:
                    need_reset = True
                elif raw_center is not None and prev_center is not None:
                    jump = np.linalg.norm(raw_center - prev_center)
                    if jump > JUMP_THRESHOLD:
                        need_reset = True

                if need_reset:
                    init_pos = raw_center if raw_center is not None else np.array([240.0, 240.0])
                    x0 = np.array([init_pos[0], init_pos[1], 0.0, 0.0, 0.0])
                    ekf = ExtendedKalmanFilter(
                        dt=dt,
                        x0=x0,
                        Q=np.diag([Q_scale, Q_scale, Q_scale * 0.5, Q_scale * 0.1, Q_scale * 0.1]),
                        R=np.eye(2) * R_scale,
                        P=np.eye(5) * 100.0,
                    )

                # EKF 적용
                ekf.predict()
                if raw_center is not None:
                    ekf.update(raw_center)
                kalman_center = ekf.get_position()

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
    parser.add_argument('--sweep', action='store_true',
                        help='다중 Q/R 값을 한번에 테스트')
    parser.add_argument('--q-values', type=str, default='0.01,0.05,0.1,0.3,0.5,1.0',
                        help='sweep 모드에서 테스트할 Q 값들 (쉼표 구분)')
    parser.add_argument('--r-values', type=str, default=None,
                        help='sweep 모드에서 테스트할 R 값들 (기본: config 값 사용)')
    parser.add_argument('--filter', type=str, default='linear',
                        choices=['linear', 'ekf', 'both'],
                        help='사용할 필터 (linear, ekf, both)')
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

    train_seq = dataset_cfg.get('train_sequences', 4)
    val_seq = dataset_cfg.get('val_sequences', 1)
    total_for_test = train_seq + val_seq + test_sequences

    print(f'\n데이터셋 로드 중...')
    print(f'  Root: {root}')
    print(f'  Test sequences: {test_sequences} (offset: {train_seq + val_seq})')

    full_dataset = UAVTrackingDataset(
        root, root_mask,
        transforms=transform,
        num_sequences=total_for_test
    )

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

    # === Sweep 모드: 다중 Q/R 테스트 ===
    if args.sweep:
        run_sweep(model, test_loader, device, kalman_cfg, args)
        return

    # === 단일 평가 실행 ===
    print(f'\n{"="*60}')
    print('Phase 3: 실제 테스트 데이터 평가')
    print(f'{"="*60}')

    # 1. Baseline (UNet only)
    print('\n[1/2] Baseline 평가 (UNet only)...')
    baseline_result = test_baseline(model, test_loader, device)

    # 2. UNet + Kalman (선형 또는 EKF)
    results_to_save = {
        'checkpoint': args.checkpoint,
        'test_samples': len(test_ds),
        'kalman_config': {
            'Q': kalman_cfg.get('process_noise'),
            'R': kalman_cfg.get('measurement_noise'),
            'dt': kalman_cfg.get('dt'),
        },
        'baseline': baseline_result,
    }

    if args.filter in ('linear', 'both'):
        print('\n[Linear KF] UNet + Linear Kalman Filter...')
        lkf_result = test_with_kalman(model, test_loader, kalman_cfg, device)
        results_to_save['linear_kalman'] = lkf_result
        print_comparison(baseline_result, lkf_result)

    if args.filter in ('ekf', 'both'):
        print('\n[EKF] UNet + Extended Kalman Filter (CTRV)...')
        ekf_result = test_with_ekf(model, test_loader, kalman_cfg, device)
        results_to_save['ekf'] = ekf_result
        print_comparison(baseline_result, ekf_result)

    if args.filter == 'both':
        # 3자 비교
        print(f'\n{"="*80}')
        print(f'{"지표":<20} {"Baseline":<12} {"Linear KF":<12} {"EKF (CTRV)":<12} {"LKF vs Base":<14} {"EKF vs Base":<14}')
        print(f'{"-"*80}')
        print(f'{"CLE (px)":<20} {baseline_result["mean_CLE_raw"]:<12.4f} '
              f'{lkf_result["mean_CLE_kalman"]:<12.4f} {ekf_result["mean_CLE_kalman"]:<12.4f} '
              f'{baseline_result["mean_CLE_raw"] - lkf_result["mean_CLE_kalman"]:+12.4f} '
              f'{baseline_result["mean_CLE_raw"] - ekf_result["mean_CLE_kalman"]:+12.4f}')
        print(f'{"Jitter":<20} {baseline_result["jitter_raw"]:<12.4f} '
              f'{lkf_result["jitter_kalman"]:<12.4f} {ekf_result["jitter_kalman"]:<12.4f} '
              f'{lkf_result["jitter_reduction"]:+12.1f}% '
              f'{ekf_result["jitter_reduction"]:+12.1f}%')
        print(f'{"Smoothness":<20} {"1.000":<12} '
              f'{lkf_result["smoothness_ratio"]:<12.3f} {ekf_result["smoothness_ratio"]:<12.3f}')
        print(f'{"-"*80}')

    # === 결과 저장 ===
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n✓ 결과 저장: {results_path}')

    if args.save_vis:
        # 가장 마지막으로 테스트한 필터 결과로 시각화
        last_result = ekf_result if args.filter in ('ekf', 'both') else lkf_result
        plot_path = save_dir / 'test_comparison.png'
        plot_test_results(baseline_result, last_result, plot_path)

    print(f'\n{"="*60}')
    print('✅ 평가 완료!')
    print(f'{"="*60}')


def run_sweep(model, test_loader, device, kalman_cfg, args):
    """다중 Q/R 파라미터 sweep 테스트"""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    q_values = [float(x) for x in args.q_values.split(',')]
    if args.r_values:
        r_values = [float(x) for x in args.r_values.split(',')]
    else:
        r_values = [kalman_cfg.get('measurement_noise', 0.5)]

    print(f'\n{"="*70}')
    print(f'  Q/R Sweep 모드')
    print(f'  Q values: {q_values}')
    print(f'  R values: {r_values}')
    print(f'{"="*70}')

    # Baseline (한번만)
    print('\n[Baseline] UNet Only...')
    baseline_result = test_baseline(model, test_loader, device)

    # 전체 결과 저장
    all_results = []

    print(f'\n{"Q":<8} {"R":<8} {"CLE":<10} {"ΔCLE":<10} {"Jitter":<10} {"Jit↓%":<10} {"Smooth":<10}')
    print('-' * 66)

    for q in q_values:
        for r in r_values:
            test_kalman_cfg = dict(kalman_cfg)
            test_kalman_cfg['process_noise'] = q
            test_kalman_cfg['measurement_noise'] = r

            result = test_with_kalman(model, test_loader, test_kalman_cfg, device)

            delta_cle = baseline_result['mean_CLE_raw'] - result['mean_CLE_kalman']

            print(f'{q:<8.3f} {r:<8.3f} {result["mean_CLE_kalman"]:<10.4f} '
                  f'{delta_cle:<+10.4f} {result["jitter_kalman"]:<10.4f} '
                  f'{result["jitter_reduction"]:<10.1f} {result["smoothness_ratio"]:<10.3f}')

            all_results.append({
                'Q': q,
                'R': r,
                'CLE_kalman': result['mean_CLE_kalman'],
                'CLE_delta': delta_cle,
                'jitter_kalman': result['jitter_kalman'],
                'jitter_reduction_pct': result['jitter_reduction'],
                'smoothness_ratio': result['smoothness_ratio'],
            })

    # 결과 저장
    sweep_output = {
        'baseline': baseline_result,
        'sweep_results': all_results,
        'q_values': q_values,
        'r_values': r_values,
    }

    sweep_path = save_dir / 'sweep_results.json'
    with open(sweep_path, 'w', encoding='utf-8') as f:
        json.dump(sweep_output, f, indent=2, ensure_ascii=False, default=str)

    # Sweep 시각화
    _plot_sweep(all_results, baseline_result, save_dir)

    # 최적 파라미터 (CLE 최소 기준)
    best = min(all_results, key=lambda x: x['CLE_kalman'])
    print(f'\n{"="*70}')
    print(f'📋 Sweep 결과 요약')
    print(f'  최적 (CLE 기준): Q={best["Q"]}, R={best["R"]}, CLE={best["CLE_kalman"]:.4f}')

    # 균형점 (CLE 증가 < 1px 중 jitter 감소 최대)
    balanced = [r for r in all_results if r['CLE_delta'] > -1.0]
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['jitter_reduction_pct'])
        print(f'  최적 (균형): Q={best_balanced["Q"]}, R={best_balanced["R"]}, '
              f'ΔCLE={best_balanced["CLE_delta"]:+.4f}, Jitter↓={best_balanced["jitter_reduction_pct"]:.1f}%')

    print(f'\n✓ 저장: {sweep_path}')
    print(f'{"="*70}')


def _plot_sweep(all_results, baseline_result, save_dir):
    """Sweep 결과 시각화"""
    q_vals = [r['Q'] for r in all_results]
    cle_vals = [r['CLE_kalman'] for r in all_results]
    jitter_vals = [r['jitter_reduction_pct'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. CLE vs Q
    axes[0].plot(q_vals, cle_vals, 'b-o', markersize=8)
    axes[0].axhline(baseline_result['mean_CLE_raw'], color='r', linestyle='--',
                   label=f'Baseline CLE={baseline_result["mean_CLE_raw"]:.2f}')
    axes[0].set_xlabel('Q (Process Noise)')
    axes[0].set_ylabel('CLE (pixels)')
    axes[0].set_title('CLE vs Q')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Jitter 감소율 vs Q
    axes[1].plot(q_vals, jitter_vals, 'g-^', markersize=8)
    axes[1].set_xlabel('Q (Process Noise)')
    axes[1].set_ylabel('Jitter Reduction (%)')
    axes[1].set_title('Jitter Reduction vs Q')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    # 3. Trade-off: CLE 증가 vs Jitter 감소
    cle_deltas = [-r['CLE_delta'] for r in all_results]  # 양수 = 나빠진 것
    axes[2].scatter(jitter_vals, cle_deltas, c=q_vals, cmap='viridis', s=100)
    for i, r in enumerate(all_results):
        axes[2].annotate(f'Q={r["Q"]}', (jitter_vals[i], cle_deltas[i]),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel('Jitter Reduction (%)')
    axes[2].set_ylabel('CLE Increase (pixels)')
    axes[2].set_title('Trade-off: CLE vs Jitter')
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'sweep_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Sweep 시각화 저장: {save_dir / "sweep_tradeoff.png"}')


def print_comparison(baseline_result, kalman_result):
    """비교 결과 출력"""
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


if __name__ == '__main__':
    main()
