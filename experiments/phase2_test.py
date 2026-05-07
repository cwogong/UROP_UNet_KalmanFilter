"""
Phase 2: UNet + Kalman Filter 통합 테스트

테스트 항목:
1. 단일 프레임 처리
2. 시퀀셜 프레임 처리 (움직이는 객체)
3. 노이즈가 있는 마스크 처리
4. 결과 시각화
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from combined_model.unet_kalman_combined import UNetKalmanCombined
from combined_model.mask_utils import MaskProcessor, TrackingVisualizer


def create_synthetic_frame_sequence(num_frames=30, image_size=480):
    """
    합성 프레임 시퀀스 생성
    
    시나리오: 원형 객체가 이미지 중앙에서 시작해 원형 경로로 움직임
    
    Args:
        num_frames (int): 생성할 프레임 수
        image_size (int): 이미지 크기
        
    Returns:
        tuple: (frames, true_masks, true_centers)
    """
    frames = []
    true_masks = []
    true_centers = []
    
    # 원형 운동 경로
    center_x, center_y = image_size // 2, image_size // 2
    radius_x = 100
    radius_y = 100
    object_radius = 30
    
    for i in range(num_frames):
        # 배경 생성 (랜덤 노이즈)
        frame = np.random.normal(0.2, 0.1, (image_size, image_size, 3))
        frame = np.clip(frame, 0, 1)
        
        # 객체 움직임 (원형 경로)
        t = 2 * np.pi * i / num_frames
        obj_x = int(center_x + radius_x * np.cos(t))
        obj_y = int(center_y + radius_y * np.sin(t))
        
        # 마스크 생성
        mask = np.zeros((image_size, image_size))
        y, x = np.ogrid[:image_size, :image_size]
        dist = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
        mask[dist <= object_radius] = 1.0
        
        # 마스크에 노이즈 추가
        noise = np.random.normal(0, 0.05, mask.shape)
        mask_noisy = np.clip(mask + noise, 0, 1)
        
        # 프레임에 객체 추가 (마스크를 흰색으로 표시)
        frame[mask > 0.5] = [1, 1, 1]
        
        frames.append(frame)
        true_masks.append(mask_noisy)
        true_centers.append(np.array([obj_x, obj_y]))
    
    # torch 텐서로 변환
    frames_tensor = [torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float() 
                     for f in frames]
    
    return frames_tensor, true_masks, true_centers


def test_single_frame():
    """테스트 1: 단일 프레임 처리"""
    print("\n" + "="*70)
    print("테스트 1: 단일 프레임 처리")
    print("="*70)
    
    # 설정
    unet_config = {
        'in_channels': 3,
        'start_out_channels': 32,
        'num_class': 1,
        'size': 4,
        'padding': 1
    }
    
    kalman_config = {
        'dt': 1.0,
        'x0': np.array([240, 240, 0, 0]),
        'Q': np.eye(4) * 0.01,
        'R': np.eye(2) * 0.5,
        'P': np.eye(4) * 100
    }
    
    # 모델 생성
    model = UNetKalmanCombined(unet_config, kalman_config, use_morphology=True)
    model.eval()
    
    # 단일 프레임 생성 및 처리
    frame = torch.randn(1, 3, 480, 480)
    
    with torch.no_grad():
        result = model(frame, use_kalman=True)
    
    print(f"\n✓ 입력 프레임 크기: {frame.shape}")
    print(f"✓ 출력 마스크 크기: {result['smoothed_mask'].shape}")
    print(f"✓ 추출된 중심점: {result['center']}")
    print(f"✓ Kalman 예측 중심점: {result['kalman_center']}")
    print(f"✓ 경계박스: {result['bbox']}")
    print(f"✓ Kalman 상태: {model.get_kalman_state()}")
    
    return model


def test_sequence_processing():
    """테스트 2: 시퀀셜 프레임 처리 (움직이는 객체)"""
    print("\n" + "="*70)
    print("테스트 2: 시퀀셜 프레임 처리 (움직이는 객체)")
    print("="*70)
    
    # 설정
    unet_config = {
        'in_channels': 3,
        'start_out_channels': 32,
        'num_class': 1,
        'size': 4,
        'padding': 1
    }
    
    kalman_config = {
        'dt': 1.0,
        'x0': np.array([240, 240, 0, 0]),
        'Q': np.eye(4) * 0.01,
        'R': np.eye(2) * 0.5,
        'P': np.eye(4) * 100
    }
    
    # 모델 생성
    model = UNetKalmanCombined(unet_config, kalman_config, use_morphology=True)
    model.eval()
    model.reset()  # 상태 초기화
    
    # 합성 데이터 생성
    print("\n📊 합성 프레임 시퀀스 생성 중...")
    frames, true_masks, true_centers = create_synthetic_frame_sequence(num_frames=30)
    print(f"✓ {len(frames)}개 프레임 생성 완료")
    
    # 시퀀셜 처리
    print("\n🔄 모델 추론 중...")
    with torch.no_grad():
        seq_result = model.process_sequence(frames, use_kalman=True)
    
    print(f"✓ 처리한 프레임: {model.get_frame_count()}")
    print(f"✓ 생성된 마스크: {len(seq_result['smoothed_masks'])}")
    print(f"✓ 추적된 중심점: {len(seq_result['centers'])}")
    
    # 메트릭 계산
    print("\n📈 추적 메트릭 계산...")
    metrics = MaskProcessor.calculate_metrics(
        seq_result['centers'], 
        seq_result['kalman_centers']
    )
    
    print(f"\n📊 결과 통계:")
    print(f"   - 처리된 프레임: {metrics.get('num_frames', 'N/A')}")
    print(f"   - 평균 거리: {metrics.get('avg_distance', 0):.2f} pixels")
    print(f"   - 최대 거리: {metrics.get('max_distance', 0):.2f} pixels")
    print(f"   - 표준편차: {metrics.get('std_distance', 0):.2f} pixels")
    print(f"   - 감지된 평균 속도: {metrics.get('detected_avg_speed', 0):.2f} pixels/frame")
    print(f"   - Kalman 평균 속도: {metrics.get('kalman_avg_speed', 0):.2f} pixels/frame")
    print(f"   - 평활도 개선: {metrics.get('smoothness_improvement', 0):.2f}%")
    
    return model, seq_result, frames, true_centers


def visualize_results(seq_result, frames, true_centers):
    """테스트 3: 결과 시각화"""
    print("\n" + "="*70)
    print("테스트 3: 결과 시각화")
    print("="*70)
    
    # 저장 디렉토리
    output_dir = Path('./experiments/phase2_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    visualizer = TrackingVisualizer(save_dir=str(output_dir))
    
    # 개별 프레임 시각화
    print("\n📸 개별 프레임 시각화 중...")
    sample_indices = [0, 7, 14, 21, 29]  # 샘플 프레임
    
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(16, 3))
    
    for idx, frame_idx in enumerate(sample_indices):
        frame_np = frames[frame_idx].squeeze().permute(1, 2, 0).numpy()
        mask_np = seq_result['smoothed_masks'][frame_idx].squeeze().numpy()
        
        vis = MaskProcessor.visualize_tracking(
            frame_np, mask_np,
            seq_result['centers'][frame_idx],
            seq_result['bboxes'][frame_idx],
            seq_result['kalman_centers'][frame_idx],
            title=f"Frame {frame_idx+1}"
        )
        
        axes[idx].imshow(vis)
        axes[idx].axis('off')
    
    plt.tight_layout()
    sample_path = output_dir / 'sample_frames.png'
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    print(f"✓ 샘플 프레임 저장: {sample_path}")
    plt.close()
    
    # 궤적 시각화
    print("\n📍 추적 궤적 시각화 중...")
    trajectory_path = output_dir / 'trajectory.png'
    MaskProcessor.plot_tracking_trajectory(
        seq_result['centers'],
        seq_result['kalman_centers'],
        save_path=trajectory_path,
        show=False
    )
    
    # 비교 그래프
    print("\n📊 비교 그래프 생성 중...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # X 축 추적
    detected_x = [c[0] if c is not None else None for c in seq_result['centers']]
    kalman_x = [k[0] if k is not None else None for k in seq_result['kalman_centers']]
    true_x = [tc[0] if tc is not None else None for tc in true_centers]
    
    frames_idx = range(len(detected_x))
    axes[0].plot(frames_idx, detected_x, 'go-', label='Detected', alpha=0.7)
    axes[0].plot(frames_idx, kalman_x, 'b^-', label='Kalman Filtered', alpha=0.7)
    axes[0].plot(frames_idx, true_x, 'r--', label='True Position', alpha=0.7)
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].set_title('Object X-Position Tracking')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y 축 추적
    detected_y = [c[1] if c is not None else None for c in seq_result['centers']]
    kalman_y = [k[1] if k is not None else None for k in seq_result['kalman_centers']]
    true_y = [tc[1] if tc is not None else None for tc in true_centers]
    
    axes[1].plot(frames_idx, detected_y, 'go-', label='Detected', alpha=0.7)
    axes[1].plot(frames_idx, kalman_y, 'b^-', label='Kalman Filtered', alpha=0.7)
    axes[1].plot(frames_idx, true_y, 'r--', label='True Position', alpha=0.7)
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].set_title('Object Y-Position Tracking')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    comparison_path = output_dir / 'position_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ 위치 비교 그래프 저장: {comparison_path}")
    plt.close()
    
    print(f"\n✅ 모든 시각화 결과가 저장되었습니다: {output_dir}")


def main():
    """메인 테스트 함수"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "Phase 2: UNet + Kalman Filter 통합 테스트" + " "*13 + "║")
    print("╚" + "="*68 + "╝")
    
    # 테스트 1: 단일 프레임
    model = test_single_frame()
    
    # 테스트 2: 시퀀셜 프레임
    model, seq_result, frames, true_centers = test_sequence_processing()
    
    # 테스트 3: 시각화
    visualize_results(seq_result, frames, true_centers)
    
    # 최종 요약
    print("\n" + "="*70)
    print("✅ Phase 2 테스트 완료!")
    print("="*70)
    print("\n🎯 다음 단계:")
    print("   1. 실제 UAV 데이터셋 준비")
    print("   2. 모델 파인 튜닝")
    print("   3. Phase 3: 실시간 추적 구현")
    print("="*70)


if __name__ == '__main__':
    main()
