"""
Phase 3: UNet vs UNet+Kalman 비교 평가

비교 대상:
1. Baseline: UNet만 사용 (raw mask → centroid)
2. Proposed: UNet + Kalman Filter (smoothed mask → filtered centroid)

평가 지표:
- mIoU, Dice (세그멘테이션 품질)
- CLE (위치 정확도)
- Jitter (추적 안정성)
- Detection Rate (추적 연속성)
- Smoothness Ratio (평활도 개선)

사용법:
    # 합성 데이터 평가 (GPU 불필요)
    python eval/evaluate.py --mode synthetic --num-frames 100

    # 체크포인트 평가 (실 데이터)
    python eval/evaluate.py --mode checkpoint --checkpoint checkpoints/demo_unet.pth --config config.yaml

    # Q/R 하이퍼파라미터 탐색
    python eval/evaluate.py --mode tuning --num-frames 100
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics import SequenceMetrics, compute_iou, compute_dice
from filters.linear_kalman_filter import KalmanFilter
from combined_model.unet_kalman_combined import UNetKalmanCombined


# ============================================================
# 합성 데이터 생성
# ============================================================

def generate_synthetic_sequence(
    num_frames: int = 100,
    image_size: int = 480,
    motion_type: str = 'circular',
    noise_level: float = 5.0,
    miss_rate: float = 0.05,
    object_radius: int = 30,
):
    """
    합성 시퀀스 생성 (GT + noisy measurement 포함)

    Args:
        num_frames: 프레임 수
        image_size: 이미지 크기
        motion_type: 'circular', 'linear', 'random_walk'
        noise_level: 측정 노이즈 표준편차 (pixels)
        miss_rate: 검출 실패 확률
        object_radius: 객체 반지름

    Returns:
        dict: {
            'frames': list of tensors (1,3,H,W),
            'gt_masks': list of np.ndarray (H,W),
            'gt_centers': list of np.ndarray [x,y],
            'noisy_centers': list of np.ndarray or None,
            'noisy_masks': list of np.ndarray (H,W),
        }
    """
    frames = []
    gt_masks = []
    gt_centers = []
    noisy_centers = []
    noisy_masks = []

    cx, cy = image_size // 2, image_size // 2

    for t in range(num_frames):
        # === Ground Truth 위치 ===
        if motion_type == 'circular':
            angle = 2 * np.pi * t / num_frames
            gt_x = cx + 120 * np.cos(angle)
            gt_y = cy + 120 * np.sin(angle)
        elif motion_type == 'linear':
            gt_x = 50 + (image_size - 100) * t / num_frames
            gt_y = cy + 30 * np.sin(4 * np.pi * t / num_frames)
        elif motion_type == 'random_walk':
            if t == 0:
                gt_x, gt_y = float(cx), float(cy)
            else:
                prev = gt_centers[-1]
                gt_x = np.clip(prev[0] + np.random.randn() * 5, 50, image_size - 50)
                gt_y = np.clip(prev[1] + np.random.randn() * 5, 50, image_size - 50)
        else:
            raise ValueError(f"Unknown motion_type: {motion_type}")

        gt_center = np.array([gt_x, gt_y], dtype=np.float32)
        gt_centers.append(gt_center)

        # === GT 마스크 ===
        y_grid, x_grid = np.ogrid[:image_size, :image_size]
        dist = np.sqrt((x_grid - gt_x) ** 2 + (y_grid - gt_y) ** 2)
        gt_mask = (dist <= object_radius).astype(np.float32)
        gt_masks.append(gt_mask)

        # === Noisy measurement (UNet 출력 시뮬레이션) ===
        if np.random.rand() < miss_rate:
            # 검출 실패
            noisy_centers.append(None)
            noisy_masks.append(np.zeros((image_size, image_size), dtype=np.float32))
        else:
            # 위치 노이즈
            noise_x = np.random.randn() * noise_level
            noise_y = np.random.randn() * noise_level
            noisy_x = gt_x + noise_x
            noisy_y = gt_y + noise_y
            noisy_centers.append(np.array([noisy_x, noisy_y], dtype=np.float32))

            # 노이즈가 포함된 마스크
            noisy_dist = np.sqrt((x_grid - noisy_x) ** 2 + (y_grid - noisy_y) ** 2)
            # 반지름에도 약간의 변동
            r_noise = object_radius + np.random.randn() * 3
            noisy_mask = (noisy_dist <= r_noise).astype(np.float32)
            noisy_masks.append(noisy_mask)

        # === 프레임 (합성 이미지) ===
        frame = np.random.normal(0.3, 0.1, (image_size, image_size, 3)).astype(np.float32)
        frame = np.clip(frame, 0, 1)
        frame[gt_mask > 0.5] = [0.9, 0.9, 0.9]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frames.append(frame_tensor)

    return {
        'frames': frames,
        'gt_masks': gt_masks,
        'gt_centers': gt_centers,
        'noisy_centers': noisy_centers,
        'noisy_masks': noisy_masks,
    }


# ============================================================
# 평가: Baseline vs Kalman
# ============================================================

def evaluate_baseline_vs_kalman(
    data: dict,
    kalman_config: dict,
    verbose: bool = True
) -> dict:
    """
    Baseline (raw detection) vs Kalman Filter 비교 평가

    Args:
        data: generate_synthetic_sequence 반환값
        kalman_config: Kalman 설정 {'dt', 'Q_scale', 'R_scale'}
        verbose: 출력 여부

    Returns:
        dict: 평가 결과
    """
    gt_masks = data['gt_masks']
    gt_centers = data['gt_centers']
    noisy_centers = data['noisy_centers']
    noisy_masks = data['noisy_masks']
    num_frames = len(gt_masks)

    # Kalman Filter 초기화
    dt = kalman_config.get('dt', 1.0)
    Q_scale = kalman_config.get('Q_scale', 0.01)
    R_scale = kalman_config.get('R_scale', 0.5)

    # 초기 위치 설정
    init_center = noisy_centers[0] if noisy_centers[0] is not None else gt_centers[0]
    x0 = np.array([init_center[0], init_center[1], 0.0, 0.0], dtype=np.float32)

    kf = KalmanFilter(
        dt=dt,
        x0=x0,
        Q=np.eye(4, dtype=np.float32) * Q_scale,
        R=np.eye(2, dtype=np.float32) * R_scale,
        P=np.eye(4, dtype=np.float32) * 100.0,
    )

    # 메트릭 수집
    baseline_metrics = SequenceMetrics()
    kalman_metrics = SequenceMetrics()

    kalman_centers_list = []

    for t in range(num_frames):
        gt_mask = gt_masks[t]
        gt_center = gt_centers[t]
        noisy_center = noisy_centers[t]
        noisy_mask = noisy_masks[t]

        # === Kalman Filter 처리 ===
        kf.predict()
        if noisy_center is not None:
            kf.update(noisy_center)
        kalman_pos = kf.get_position()
        kalman_centers_list.append(kalman_pos.copy())

        # === Baseline 평가 (raw detection) ===
        baseline_metrics.update(
            pred_mask=noisy_mask,
            gt_mask=gt_mask,
            raw_center=noisy_center,
            gt_center=gt_center,
            kalman_center=noisy_center,  # baseline은 필터 미적용
        )

        # === Kalman 평가 ===
        kalman_metrics.update(
            pred_mask=noisy_mask,  # 마스크는 동일 (Kalman은 위치만 보정)
            gt_mask=gt_mask,
            raw_center=noisy_center,
            gt_center=gt_center,
            kalman_center=kalman_pos,
        )

    baseline_result = baseline_metrics.summarize()
    kalman_result = kalman_metrics.summarize()

    if verbose:
        print("\n" + "=" * 70)
        print("📊 Phase 3: Baseline vs Kalman Filter 비교 평가")
        print("=" * 70)
        print(f"\n{'지표':<25} {'Baseline':<15} {'Kalman':<15} {'개선':<15}")
        print("-" * 70)
        print(f"{'mIoU':<25} {baseline_result['mIoU']:<15.4f} {kalman_result['mIoU']:<15.4f} {'(동일)':<15}")
        print(f"{'Dice':<25} {baseline_result['mean_dice']:<15.4f} {kalman_result['mean_dice']:<15.4f} {'(동일)':<15}")
        print(f"{'CLE (pixels)':<25} {baseline_result['mean_CLE_raw']:<15.4f} {kalman_result['mean_CLE_kalman']:<15.4f} {baseline_result['mean_CLE_raw'] - kalman_result['mean_CLE_kalman']:+.4f}")
        print(f"{'Jitter (px/frame²)':<25} {baseline_result['jitter_raw']:<15.4f} {kalman_result['jitter_kalman']:<15.4f} {kalman_result['jitter_reduction']:+.1f}%")
        print(f"{'Detection Rate':<25} {baseline_result['detection_rate_raw']:<15.4f} {kalman_result['detection_rate_kalman']:<15.4f} {'---':<15}")
        print(f"{'Smoothness Ratio':<25} {'---':<15} {kalman_result['smoothness_ratio']:<15.4f} {'(>1=개선)':<15}")
        print("-" * 70)

    return {
        'baseline': baseline_result,
        'kalman': kalman_result,
        'kalman_centers': kalman_centers_list,
    }


# ============================================================
# Q/R 하이퍼파라미터 튜닝
# ============================================================

def tune_qr_parameters(
    data: dict,
    q_range: list = None,
    r_range: list = None,
    verbose: bool = True,
) -> dict:
    """
    Q (process noise), R (measurement noise) 그리드 서치

    Args:
        data: 합성 데이터
        q_range: Q 스케일 값 리스트
        r_range: R 스케일 값 리스트

    Returns:
        dict: 최적 파라미터 및 전체 결과
    """
    if q_range is None:
        q_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    if r_range is None:
        r_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    best_cle = float('inf')
    best_params = None

    total = len(q_range) * len(r_range)
    count = 0

    for q_scale in q_range:
        for r_scale in r_range:
            count += 1
            kalman_config = {'dt': 1.0, 'Q_scale': q_scale, 'R_scale': r_scale}
            eval_result = evaluate_baseline_vs_kalman(data, kalman_config, verbose=False)

            cle = eval_result['kalman']['mean_CLE_kalman']
            jitter = eval_result['kalman']['jitter_kalman']

            results.append({
                'Q_scale': q_scale,
                'R_scale': r_scale,
                'CLE': cle,
                'jitter': jitter,
                'smoothness_ratio': eval_result['kalman']['smoothness_ratio'],
            })

            if cle < best_cle:
                best_cle = cle
                best_params = {'Q_scale': q_scale, 'R_scale': r_scale}

            if verbose and count % 10 == 0:
                print(f"  진행: {count}/{total}")

    if verbose:
        print("\n" + "=" * 70)
        print("🔧 Q/R 하이퍼파라미터 튜닝 결과")
        print("=" * 70)
        print(f"\n최적 파라미터: Q_scale={best_params['Q_scale']}, R_scale={best_params['R_scale']}")
        print(f"최소 CLE: {best_cle:.4f} pixels")
        print(f"\n상위 5개 조합:")
        sorted_results = sorted(results, key=lambda x: x['CLE'])
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. Q={r['Q_scale']:<8} R={r['R_scale']:<8} "
                  f"CLE={r['CLE']:.4f}  Jitter={r['jitter']:.4f}  "
                  f"Smoothness={r['smoothness_ratio']:.2f}")

    return {
        'best_params': best_params,
        'best_cle': best_cle,
        'all_results': results,
    }


# ============================================================
# 시각화
# ============================================================

def plot_evaluation_results(
    data: dict,
    eval_result: dict,
    save_dir: str = 'eval/results',
):
    """평가 결과 시각화 및 저장"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    gt_centers = np.array(data['gt_centers'])
    noisy_centers_raw = data['noisy_centers']
    kalman_centers = np.array(eval_result['kalman_centers'])

    # None을 NaN으로 변환
    noisy_arr = np.full((len(noisy_centers_raw), 2), np.nan)
    for i, c in enumerate(noisy_centers_raw):
        if c is not None:
            noisy_arr[i] = c

    num_frames = len(gt_centers)
    frame_idx = np.arange(num_frames)

    # === Figure 1: 궤적 비교 ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(gt_centers[:, 0], gt_centers[:, 1], 'g-', linewidth=2, label='Ground Truth')
    ax.plot(noisy_arr[:, 0], noisy_arr[:, 1], 'r.', markersize=4, alpha=0.5, label='Raw Detection')
    ax.plot(kalman_centers[:, 0], kalman_centers[:, 1], 'b-', linewidth=1.5, label='Kalman Filtered')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Trajectory Comparison: GT vs Raw vs Kalman')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path / 'trajectory_comparison.png', dpi=150)
    plt.close()

    # === Figure 2: X/Y 위치 시계열 ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(frame_idx, gt_centers[:, 0], 'g-', linewidth=2, label='GT')
    axes[0].plot(frame_idx, noisy_arr[:, 0], 'r.', markersize=3, alpha=0.5, label='Raw')
    axes[0].plot(frame_idx, kalman_centers[:, 0], 'b-', linewidth=1.5, label='Kalman')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].set_title('X-Position Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frame_idx, gt_centers[:, 1], 'g-', linewidth=2, label='GT')
    axes[1].plot(frame_idx, noisy_arr[:, 1], 'r.', markersize=3, alpha=0.5, label='Raw')
    axes[1].plot(frame_idx, kalman_centers[:, 1], 'b-', linewidth=1.5, label='Kalman')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].set_title('Y-Position Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'position_timeseries.png', dpi=150)
    plt.close()

    # === Figure 3: CLE 시계열 ===
    cle_raw = []
    cle_kalman = []
    for t in range(num_frames):
        gt = gt_centers[t]
        raw = noisy_centers_raw[t]
        kal = kalman_centers[t]

        if raw is not None:
            cle_raw.append(np.linalg.norm(raw - gt))
        else:
            cle_raw.append(np.nan)
        cle_kalman.append(np.linalg.norm(kal - gt))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame_idx, cle_raw, 'r-', alpha=0.6, label=f'Raw CLE (mean={np.nanmean(cle_raw):.2f})')
    ax.plot(frame_idx, cle_kalman, 'b-', alpha=0.8, label=f'Kalman CLE (mean={np.mean(cle_kalman):.2f})')
    ax.axhline(np.nanmean(cle_raw), color='r', linestyle='--', alpha=0.4)
    ax.axhline(np.mean(cle_kalman), color='b', linestyle='--', alpha=0.4)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Center Location Error (pixels)')
    ax.set_title('CLE Comparison: Raw Detection vs Kalman Filter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'cle_comparison.png', dpi=150)
    plt.close()

    print(f"\n✓ 시각화 결과 저장: {save_path}/")
    print(f"  - trajectory_comparison.png")
    print(f"  - position_timeseries.png")
    print(f"  - cle_comparison.png")


def plot_tuning_heatmap(tuning_result: dict, save_dir: str = 'eval/results'):
    """Q/R 튜닝 결과 히트맵"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = tuning_result['all_results']

    q_values = sorted(set(r['Q_scale'] for r in results))
    r_values = sorted(set(r['R_scale'] for r in results))

    # CLE 히트맵 데이터
    cle_grid = np.zeros((len(q_values), len(r_values)))
    jitter_grid = np.zeros((len(q_values), len(r_values)))

    for r in results:
        qi = q_values.index(r['Q_scale'])
        ri = r_values.index(r['R_scale'])
        cle_grid[qi, ri] = r['CLE']
        jitter_grid[qi, ri] = r['jitter']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CLE 히트맵
    im1 = axes[0].imshow(cle_grid, aspect='auto', cmap='RdYlGn_r')
    axes[0].set_xticks(range(len(r_values)))
    axes[0].set_xticklabels([f'{v}' for v in r_values])
    axes[0].set_yticks(range(len(q_values)))
    axes[0].set_yticklabels([f'{v}' for v in q_values])
    axes[0].set_xlabel('R (Measurement Noise)')
    axes[0].set_ylabel('Q (Process Noise)')
    axes[0].set_title('CLE (lower = better)')
    plt.colorbar(im1, ax=axes[0])

    # 최적점 표시
    best = tuning_result['best_params']
    best_qi = q_values.index(best['Q_scale'])
    best_ri = r_values.index(best['R_scale'])
    axes[0].plot(best_ri, best_qi, 'r*', markersize=20)

    # Jitter 히트맵
    im2 = axes[1].imshow(jitter_grid, aspect='auto', cmap='RdYlGn_r')
    axes[1].set_xticks(range(len(r_values)))
    axes[1].set_xticklabels([f'{v}' for v in r_values])
    axes[1].set_yticks(range(len(q_values)))
    axes[1].set_yticklabels([f'{v}' for v in q_values])
    axes[1].set_xlabel('R (Measurement Noise)')
    axes[1].set_ylabel('Q (Process Noise)')
    axes[1].set_title('Jitter (lower = better)')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path / 'qr_tuning_heatmap.png', dpi=150)
    plt.close()

    print(f"✓ 튜닝 히트맵 저장: {save_path / 'qr_tuning_heatmap.png'}")


# ============================================================
# 메인
# ============================================================

def run_synthetic_evaluation(num_frames: int = 100, noise_level: float = 5.0):
    """합성 데이터 기반 전체 평가 파이프라인"""
    print("\n╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Phase 3: 성능 평가 (합성 데이터)" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. 합성 데이터 생성
    print("\n[1/4] 합성 데이터 생성...")
    motion_types = ['circular', 'linear', 'random_walk']
    all_results = {}

    for motion in motion_types:
        print(f"\n  --- 모션 타입: {motion} ---")
        data = generate_synthetic_sequence(
            num_frames=num_frames,
            image_size=480,
            motion_type=motion,
            noise_level=noise_level,
            miss_rate=0.05,
        )

        # 2. Baseline vs Kalman 비교
        kalman_config = {'dt': 1.0, 'Q_scale': 0.01, 'R_scale': 0.5}
        eval_result = evaluate_baseline_vs_kalman(data, kalman_config, verbose=True)

        all_results[motion] = {
            'data': data,
            'eval_result': eval_result,
        }

    # 3. 시각화 (circular 모션 기준)
    print("\n[3/4] 시각화 생성...")
    plot_evaluation_results(
        all_results['circular']['data'],
        all_results['circular']['eval_result'],
        save_dir='eval/results',
    )

    # 4. Q/R 튜닝
    print("\n[4/4] Q/R 하이퍼파라미터 탐색...")
    tuning_data = generate_synthetic_sequence(
        num_frames=num_frames,
        image_size=480,
        motion_type='circular',
        noise_level=noise_level,
    )
    tuning_result = tune_qr_parameters(tuning_data, verbose=True)
    plot_tuning_heatmap(tuning_result, save_dir='eval/results')

    # 5. 결과 저장
    summary = {
        'num_frames': num_frames,
        'noise_level': noise_level,
        'best_qr_params': tuning_result['best_params'],
        'best_cle': tuning_result['best_cle'],
        'motion_results': {},
    }
    for motion in motion_types:
        res = all_results[motion]['eval_result']
        summary['motion_results'][motion] = {
            'baseline': res['baseline'],
            'kalman': res['kalman'],
        }

    results_path = Path('eval/results/evaluation_summary.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ 평가 요약 저장: {results_path}")

    # 최종 요약
    print("\n" + "=" * 70)
    print("✅ Phase 3 평가 완료!")
    print("=" * 70)
    print(f"\n📋 최적 Kalman 파라미터:")
    print(f"   Q (process noise) = {tuning_result['best_params']['Q_scale']}")
    print(f"   R (measurement noise) = {tuning_result['best_params']['R_scale']}")
    print(f"   최소 CLE = {tuning_result['best_cle']:.4f} pixels")
    print("=" * 70)

    return summary


def run_checkpoint_evaluation(checkpoint_path: str, config_path: str, num_frames: int = 50):
    """체크포인트 기반 평가 (실제 UNet 추론 사용)"""
    import yaml

    print("\n╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Phase 3: 체크포인트 평가" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")

    # config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get('model', {})
    kalman_cfg = cfg.get('kalman', {})

    # UNet + Kalman 통합 모델 로드
    unet_config = {
        'in_channels': model_cfg.get('in_channels', 3),
        'start_out_channels': model_cfg.get('start_out_channels', 32),
        'num_class': model_cfg.get('num_class', 1),
        'size': model_cfg.get('size', 4),
        'padding': model_cfg.get('padding', 1),
    }

    x0 = np.array(kalman_cfg.get('x0', [240, 240, 0, 0]), dtype=np.float32)
    kalman_config = {
        'dt': kalman_cfg.get('dt', 1.0),
        'x0': x0,
        'Q': np.eye(4, dtype=np.float32) * kalman_cfg.get('process_noise', 0.01),
        'R': np.eye(2, dtype=np.float32) * kalman_cfg.get('measurement_noise', 0.5),
        'P': np.eye(4, dtype=np.float32) * 100.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 로드
    model = UNetKalmanCombined(unet_config, kalman_config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)

    # UNet 부분만 로드 (통합 모델의 unet 서브모듈)
    unet_state = {}
    for k, v in state_dict.items():
        if k.startswith('unet.'):
            unet_state[k[5:]] = v
        elif k.startswith('encoder.') or k.startswith('decoder.'):
            unet_state[k] = v
        else:
            unet_state[k] = v

    try:
        model.unet.load_state_dict(unet_state, strict=True)
    except RuntimeError:
        model.unet.load_state_dict(unet_state, strict=False)
        print("⚠️ 일부 가중치 불일치 (strict=False)")

    model.to(device)
    model.eval()
    print(f"✓ 체크포인트 로드 완료: {checkpoint_path}")

    # 합성 데이터로 평가 (실 데이터셋 경로가 없을 수 있으므로)
    print("\n합성 시퀀스로 UNet+Kalman 추론 평가...")
    data = generate_synthetic_sequence(num_frames=num_frames, motion_type='circular', noise_level=5.0)

    metrics_with_kalman = SequenceMetrics()
    metrics_without_kalman = SequenceMetrics()

    model.reset()

    with torch.no_grad():
        for t in range(num_frames):
            frame = data['frames'][t].to(device)
            gt_mask = data['gt_masks'][t]
            gt_center = data['gt_centers'][t]

            # With Kalman
            result = model(frame, use_kalman=True)
            pred_mask = result['smoothed_mask'].squeeze().cpu().numpy()
            raw_center = result['center']
            kalman_center = result['kalman_center']

            metrics_with_kalman.update(pred_mask, gt_mask, raw_center, gt_center, kalman_center)

    # Without Kalman (리셋 후 재실행)
    model.reset()
    with torch.no_grad():
        for t in range(num_frames):
            frame = data['frames'][t].to(device)
            gt_mask = data['gt_masks'][t]
            gt_center = data['gt_centers'][t]

            result = model(frame, use_kalman=False)
            pred_mask = result['smoothed_mask'].squeeze().cpu().numpy()
            raw_center = result['center']

            metrics_without_kalman.update(pred_mask, gt_mask, raw_center, gt_center, raw_center)

    # 결과 비교
    res_with = metrics_with_kalman.summarize()
    res_without = metrics_without_kalman.summarize()

    print("\n" + "=" * 70)
    print(f"{'지표':<25} {'UNet Only':<15} {'UNet+Kalman':<15} {'개선':<15}")
    print("-" * 70)
    print(f"{'mIoU':<25} {res_without['mIoU']:<15.4f} {res_with['mIoU']:<15.4f} {res_with['mIoU'] - res_without['mIoU']:+.4f}")
    print(f"{'Dice':<25} {res_without['mean_dice']:<15.4f} {res_with['mean_dice']:<15.4f} {res_with['mean_dice'] - res_without['mean_dice']:+.4f}")
    print(f"{'CLE (pixels)':<25} {res_without['mean_CLE_raw']:<15.4f} {res_with['mean_CLE_kalman']:<15.4f} {res_without['mean_CLE_raw'] - res_with['mean_CLE_kalman']:+.4f}")
    print(f"{'Jitter':<25} {res_without['jitter_raw']:<15.4f} {res_with['jitter_kalman']:<15.4f} {res_with['jitter_reduction']:+.1f}%")
    print(f"{'Detection Rate':<25} {res_without['detection_rate_raw']:<15.4f} {res_with['detection_rate_kalman']:<15.4f}")
    print("-" * 70)

    return {'unet_only': res_without, 'unet_kalman': res_with}


def main():
    parser = argparse.ArgumentParser(description='Phase 3: 평가')
    parser.add_argument('--mode', choices=['synthetic', 'checkpoint', 'tuning'],
                        default='synthetic', help='평가 모드')
    parser.add_argument('--num-frames', type=int, default=100, help='프레임 수')
    parser.add_argument('--noise-level', type=float, default=5.0, help='측정 노이즈 수준')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/demo_unet.pth')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    if args.mode == 'synthetic':
        run_synthetic_evaluation(num_frames=args.num_frames, noise_level=args.noise_level)
    elif args.mode == 'checkpoint':
        run_checkpoint_evaluation(args.checkpoint, args.config, num_frames=args.num_frames)
    elif args.mode == 'tuning':
        print("\n[Q/R 튜닝 모드]")
        data = generate_synthetic_sequence(
            num_frames=args.num_frames,
            motion_type='circular',
            noise_level=args.noise_level,
        )
        tuning_result = tune_qr_parameters(data, verbose=True)
        plot_tuning_heatmap(tuning_result, save_dir='eval/results')


if __name__ == '__main__':
    main()
