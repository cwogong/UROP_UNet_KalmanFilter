"""
Phase 3: 노이즈 민감도 분석

다양한 노이즈 수준에서 Kalman Filter의 효과를 정량적으로 측정.
목적: Kalman Filter가 언제 효과적이고, 언제 한계를 보이는지 분석.

사용법:
    python eval/noise_sensitivity.py
    python eval/noise_sensitivity.py --num-frames 200 --motion circular
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from eval.evaluate import generate_synthetic_sequence, evaluate_baseline_vs_kalman


def noise_sensitivity_analysis(
    noise_levels: list = None,
    num_frames: int = 100,
    motion_type: str = 'circular',
    kalman_config: dict = None,
    save_dir: str = 'eval/results',
):
    """
    노이즈 수준별 성능 분석

    Args:
        noise_levels: 테스트할 노이즈 수준 리스트 (pixels)
        num_frames: 시퀀스 길이
        motion_type: 모션 유형
        kalman_config: Kalman 설정
        save_dir: 저장 경로
    """
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    if kalman_config is None:
        kalman_config = {'dt': 1.0, 'Q_scale': 0.01, 'R_scale': 0.5}

    print("\n╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Phase 3: 노이즈 민감도 분석" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n모션: {motion_type}, 프레임: {num_frames}")
    print(f"노이즈 수준: {noise_levels}")
    print(f"Kalman 설정: Q={kalman_config['Q_scale']}, R={kalman_config['R_scale']}")

    results = {
        'noise_levels': noise_levels,
        'cle_baseline': [],
        'cle_kalman': [],
        'jitter_baseline': [],
        'jitter_kalman': [],
        'smoothness_ratio': [],
        'cle_improvement_pct': [],
    }

    print(f"\n{'Noise':<10} {'CLE(Raw)':<12} {'CLE(Kalman)':<12} {'개선%':<10} "
          f"{'Jitter(R)':<12} {'Jitter(K)':<12} {'Smooth':<10}")
    print("-" * 80)

    for noise in noise_levels:
        # 데이터 생성
        data = generate_synthetic_sequence(
            num_frames=num_frames,
            image_size=480,
            motion_type=motion_type,
            noise_level=noise,
            miss_rate=0.03,
        )

        # 평가
        eval_result = evaluate_baseline_vs_kalman(data, kalman_config, verbose=False)
        baseline = eval_result['baseline']
        kalman = eval_result['kalman']

        cle_raw = baseline['mean_CLE_raw']
        cle_kal = kalman['mean_CLE_kalman']
        jit_raw = baseline['jitter_raw']
        jit_kal = kalman['jitter_kalman']
        smooth = kalman['smoothness_ratio']
        improvement = (cle_raw - cle_kal) / (cle_raw + 1e-8) * 100

        results['cle_baseline'].append(cle_raw)
        results['cle_kalman'].append(cle_kal)
        results['jitter_baseline'].append(jit_raw)
        results['jitter_kalman'].append(jit_kal)
        results['smoothness_ratio'].append(smooth)
        results['cle_improvement_pct'].append(improvement)

        print(f"{noise:<10.1f} {cle_raw:<12.4f} {cle_kal:<12.4f} {improvement:<10.1f} "
              f"{jit_raw:<12.4f} {jit_kal:<12.4f} {smooth:<10.2f}")

    # === 시각화 ===
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CLE 비교
    axes[0, 0].plot(noise_levels, results['cle_baseline'], 'r-o', label='Baseline (Raw)')
    axes[0, 0].plot(noise_levels, results['cle_kalman'], 'b-^', label='Kalman Filter')
    axes[0, 0].set_xlabel('Measurement Noise (pixels)')
    axes[0, 0].set_ylabel('CLE (pixels)')
    axes[0, 0].set_title('Center Location Error vs Noise Level')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. CLE 개선율
    axes[0, 1].bar(range(len(noise_levels)), results['cle_improvement_pct'],
                   color='green', alpha=0.7)
    axes[0, 1].set_xticks(range(len(noise_levels)))
    axes[0, 1].set_xticklabels([f'{n}' for n in noise_levels])
    axes[0, 1].set_xlabel('Measurement Noise (pixels)')
    axes[0, 1].set_ylabel('CLE Improvement (%)')
    axes[0, 1].set_title('Kalman Filter CLE Improvement')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)

    # 3. Jitter 비교
    axes[1, 0].plot(noise_levels, results['jitter_baseline'], 'r-o', label='Baseline')
    axes[1, 0].plot(noise_levels, results['jitter_kalman'], 'b-^', label='Kalman')
    axes[1, 0].set_xlabel('Measurement Noise (pixels)')
    axes[1, 0].set_ylabel('Jitter (pixels/frame²)')
    axes[1, 0].set_title('Tracking Jitter vs Noise Level')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Smoothness Ratio
    axes[1, 1].plot(noise_levels, results['smoothness_ratio'], 'g-s', linewidth=2)
    axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No improvement')
    axes[1, 1].set_xlabel('Measurement Noise (pixels)')
    axes[1, 1].set_ylabel('Smoothness Ratio')
    axes[1, 1].set_title('Smoothness Ratio (>1 = Kalman is smoother)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Noise Sensitivity Analysis (motion={motion_type})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / 'noise_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 결과 저장
    results_file = save_path / 'noise_sensitivity.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 시각화 저장: {save_path / 'noise_sensitivity.png'}")
    print(f"✓ 데이터 저장: {results_file}")

    # 요약
    print("\n" + "=" * 70)
    print("📋 노이즈 민감도 분석 요약")
    print("=" * 70)
    avg_improvement = np.mean(results['cle_improvement_pct'])
    max_improvement_idx = np.argmax(results['cle_improvement_pct'])
    print(f"  평균 CLE 개선: {avg_improvement:.1f}%")
    print(f"  최대 개선 노이즈: {noise_levels[max_improvement_idx]:.1f} pixels "
          f"({results['cle_improvement_pct'][max_improvement_idx]:.1f}%)")
    print(f"  평균 Smoothness Ratio: {np.mean(results['smoothness_ratio']):.2f}")

    if avg_improvement > 0:
        print("\n  → Kalman Filter가 전반적으로 추적 정확도를 개선함")
    else:
        print("\n  → 현재 Q/R 설정에서는 Kalman Filter 효과가 미미함. 튜닝 필요.")

    return results


def miss_rate_analysis(
    miss_rates: list = None,
    num_frames: int = 100,
    noise_level: float = 5.0,
    motion_type: str = 'circular',
    save_dir: str = 'eval/results',
):
    """
    검출 실패율에 따른 Kalman Filter 효과 분석

    Kalman Filter의 핵심 장점: 검출 실패 시에도 predict만으로 위치 유지
    """
    if miss_rates is None:
        miss_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    kalman_config = {'dt': 1.0, 'Q_scale': 0.01, 'R_scale': 0.5}

    print("\n" + "=" * 70)
    print("📊 검출 실패율 분석")
    print("=" * 70)
    print(f"\n{'Miss Rate':<12} {'CLE(Raw)':<12} {'CLE(Kalman)':<12} "
          f"{'DetRate(R)':<12} {'DetRate(K)':<12}")
    print("-" * 60)

    results = {
        'miss_rates': miss_rates,
        'cle_baseline': [],
        'cle_kalman': [],
        'det_rate_raw': [],
        'det_rate_kalman': [],
    }

    for miss_rate in miss_rates:
        data = generate_synthetic_sequence(
            num_frames=num_frames,
            motion_type=motion_type,
            noise_level=noise_level,
            miss_rate=miss_rate,
        )

        eval_result = evaluate_baseline_vs_kalman(data, kalman_config, verbose=False)
        baseline = eval_result['baseline']
        kalman = eval_result['kalman']

        results['cle_baseline'].append(baseline['mean_CLE_raw'])
        results['cle_kalman'].append(kalman['mean_CLE_kalman'])
        results['det_rate_raw'].append(baseline['detection_rate_raw'])
        results['det_rate_kalman'].append(kalman['detection_rate_kalman'])

        print(f"{miss_rate:<12.2f} {baseline['mean_CLE_raw']:<12.4f} "
              f"{kalman['mean_CLE_kalman']:<12.4f} "
              f"{baseline['detection_rate_raw']:<12.4f} "
              f"{kalman['detection_rate_kalman']:<12.4f}")

    # 시각화
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(miss_rates, results['cle_baseline'], 'r-o', label='Baseline')
    axes[0].plot(miss_rates, results['cle_kalman'], 'b-^', label='Kalman')
    axes[0].set_xlabel('Detection Miss Rate')
    axes[0].set_ylabel('CLE (pixels)')
    axes[0].set_title('CLE vs Detection Miss Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(miss_rates, results['det_rate_raw'], 'r-o', label='Raw Detection Rate')
    axes[1].plot(miss_rates, results['det_rate_kalman'], 'b-^', label='Kalman (always predicts)')
    axes[1].set_xlabel('Detection Miss Rate')
    axes[1].set_ylabel('Effective Detection Rate')
    axes[1].set_title('Tracking Continuity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'miss_rate_analysis.png', dpi=150)
    plt.close()

    print(f"\n✓ 저장: {save_path / 'miss_rate_analysis.png'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--motion', type=str, default='circular',
                        choices=['circular', 'linear', 'random_walk'])
    parser.add_argument('--save-dir', type=str, default='eval/results')
    args = parser.parse_args()

    # 노이즈 민감도
    noise_sensitivity_analysis(
        num_frames=args.num_frames,
        motion_type=args.motion,
        save_dir=args.save_dir,
    )

    # 검출 실패율 분석
    miss_rate_analysis(
        num_frames=args.num_frames,
        motion_type=args.motion,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
