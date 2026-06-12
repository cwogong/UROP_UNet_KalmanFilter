"""
Phase 3: 전체 평가 파이프라인 실행

실행:
    python eval/run_phase3.py

출력:
    eval/results/
    ├── trajectory_comparison.png     # 궤적 비교 (GT vs Raw vs Kalman)
    ├── position_timeseries.png       # X/Y 위치 시계열
    ├── cle_comparison.png            # CLE 비교 그래프
    ├── qr_tuning_heatmap.png         # Q/R 튜닝 히트맵
    ├── noise_sensitivity.png         # 노이즈 민감도 그래프
    ├── miss_rate_analysis.png        # 검출 실패율 분석
    ├── evaluation_summary.json       # 수치 결과 요약
    └── noise_sensitivity.json        # 노이즈 민감도 수치 데이터
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from eval.evaluate import run_synthetic_evaluation
from eval.noise_sensitivity import noise_sensitivity_analysis, miss_rate_analysis


def main():
    print("=" * 70)
    print("  Phase 3: 전체 평가 파이프라인")
    print("  - Baseline (UNet only) vs Proposed (UNet + Kalman Filter)")
    print("=" * 70)

    # === Step 1: 종합 평가 (3가지 모션 타입) ===
    summary = run_synthetic_evaluation(num_frames=100, noise_level=5.0)

    # === Step 2: 노이즈 민감도 분석 ===
    print("\n\n")
    noise_sensitivity_analysis(
        num_frames=100,
        motion_type='circular',
        save_dir='eval/results',
    )

    # === Step 3: 검출 실패율 분석 ===
    print("\n\n")
    miss_rate_analysis(
        num_frames=100,
        motion_type='circular',
        save_dir='eval/results',
    )

    # === 최종 정리 ===
    print("\n\n" + "=" * 70)
    print("✅ Phase 3 전체 평가 완료!")
    print("=" * 70)
    print("\n📁 결과 디렉토리: eval/results/")
    print("\n📊 핵심 결론:")

    best_q = summary.get('best_qr_params', {}).get('Q_scale', 'N/A')
    best_r = summary.get('best_qr_params', {}).get('R_scale', 'N/A')
    best_cle = summary.get('best_cle', 'N/A')

    print(f"   - 최적 Q (process noise): {best_q}")
    print(f"   - 최적 R (measurement noise): {best_r}")
    print(f"   - 최소 CLE: {best_cle}")
    print(f"\n🔬 다음 단계:")
    print(f"   - config.yaml의 kalman.process_noise를 {best_q}로 업데이트")
    print(f"   - config.yaml의 kalman.measurement_noise를 {best_r}로 업데이트")
    print(f"   - Phase 4: EKF/UKF 비선형 필터로 확장")
    print("=" * 70)


if __name__ == '__main__':
    main()
