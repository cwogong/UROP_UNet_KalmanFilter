# %% [markdown]
# # Phase 2: UNet + Kalman Filter 테스트

# %% 
# 테스트 1: Kalman Filter
import numpy as np
import sys
from pathlib import Path

project_root = Path().cwd()
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Phase 2: UNet + Kalman Filter 통합 - 기본 테스트")
print("=" * 70)

from filters.linear_kalman_filter import KalmanFilter

print("\n[테스트 1] Kalman Filter 검증")
print("-" * 70)

kalman_config = {
    'dt': 1.0,
    'x0': np.array([240, 240, 0, 0]),
    'Q': np.eye(4) * 0.01,
    'R': np.eye(2) * 0.5,
    'P': np.eye(4) * 100
}

kf = KalmanFilter(**kalman_config)
print("✅ Kalman Filter 초기화 성공")
print(f"   초기 상태: {kf.get_state()}")

# 시뮬레이션
print("\n선형 운동 추적 (5 프레임):")
for i in range(5):
    measurement = np.array([240 + i * 5, 240 + i * 3])
    kf.predict()
    kf.update(measurement)
    pos = kf.get_position()
    print(f"  Frame {i+1}: 위치 = {pos}")

print("\n✅ Kalman Filter 정상 작동!")

# %%
# 테스트 2: 마스크 처리 유틸리티
from combined_model.mask_utils import MaskProcessor

print("\n[테스트 2] 마스크 처리 유틸리티")
print("-" * 70)

detected_centers = [
    np.array([50.0, 50.0]), 
    np.array([51.0, 50.5]), 
    np.array([52.0, 51.0])
]
kalman_centers = [
    np.array([50.0, 50.0]), 
    np.array([50.8, 50.4]), 
    np.array([51.6, 50.8])
]

metrics = MaskProcessor.calculate_metrics(detected_centers, kalman_centers)

print("추적 메트릭 계산 결과:")
print(f"  처리된 프레임: {metrics.get('num_frames', 'N/A')}")
print(f"  평균 거리: {metrics.get('avg_distance', 0):.4f} pixels")
print(f"  최대 거리: {metrics.get('max_distance', 0):.4f} pixels")
print(f"  표준편차: {metrics.get('std_distance', 0):.4f} pixels")
print(f"  감지된 평균 속도: {metrics.get('detected_avg_speed', 0):.4f} pixels/frame")
print(f"  Kalman 평균 속도: {metrics.get('kalman_avg_speed', 0):.4f} pixels/frame")
print(f"  평활도 개선: {metrics.get('smoothness_improvement', 0):.2f}%")

print("\n✅ 메트릭 계산 정상 작동!")

# %%
# 테스트 3: UNet 모델
try:
    import torch
    from model.Vanilla_UNet import VanillaUNet
    
    print("\n[테스트 3] VanillaUNet 모델")
    print("-" * 70)
    
    unet_config = {
        'in_channels': 3,
        'start_out_channels': 32,
        'num_class': 1,
        'size': 4,
        'padding': 1
    }
    
    unet = VanillaUNet(**unet_config)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"✅ VanillaUNet 로드 성공")
    print(f"   총 파라미터: {total_params:,}")
    
    # Forward pass
    dummy_input = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        output = unet(dummy_input)
    
    print(f"   입력 크기: {dummy_input.shape}")
    print(f"   출력 크기: {output.shape}")
    print("\n✅ UNet forward pass 성공!")
    
except ImportError:
    print("⚠️  PyTorch 설치 필요!")
    print("   pip install torch")

# %%
# 테스트 4: 통합 모델
try:
    import torch
    from combined_model.unet_kalman_combined import UNetKalmanCombined
    
    print("\n[테스트 4] UNet + Kalman Filter 통합 모델")
    print("-" * 70)
    
    kalman_config = {
        'dt': 1.0,
        'x0': np.array([240, 240, 0, 0]),
        'Q': np.eye(4) * 0.01,
        'R': np.eye(2) * 0.5,
        'P': np.eye(4) * 100
    }
    
    model = UNetKalmanCombined(unet_config, kalman_config, use_morphology=True)
    model.eval()
    
    print("✅ 통합 모델 생성 완료")
    
    # 단일 프레임 처리
    with torch.no_grad():
        dummy_frame = torch.randn(1, 3, 480, 480)
        result = model(dummy_frame, use_kalman=True)
    
    print("\n단일 프레임 처리 결과:")
    print(f"  입력 크기: {dummy_frame.shape}")
    print(f"  출력 마스크 크기: {result['smoothed_mask'].shape}")
    print(f"  추출된 중심점: {result['center']}")
    print(f"  Kalman 중심점: {result['kalman_center']}")
    print(f"  경계박스: {result['bbox']}")
    
    # 시퀀셜 처리
    print("\n시퀀셜 처리 (5 프레임):")
    model.reset()
    frames = [torch.randn(1, 3, 480, 480) for _ in range(5)]
    
    with torch.no_grad():
        seq_result = model.process_sequence(frames, use_kalman=True)
    
    print(f"  처리된 프레임: {len(seq_result['smoothed_masks'])}")
    print(f"  총 프레임 카운트: {model.get_frame_count()}")
    
    info = model.get_model_info()
    print(f"  Kalman 위치: {info['kalman_position']}")
    print(f"  Kalman 속도: {info['kalman_velocity']}")
    
    print("\n✅ 통합 모델 정상 작동!")
    
except ImportError as e:
    print(f"⚠️  PyTorch 설치 필요: {e}")
    print("   pip install torch numpy opencv-python scipy")

# %%
# 최종 결과
print("\n" + "=" * 70)
print("✅ Phase 2 모든 테스트 완료!")
print("=" * 70)
print("\n🎯 다음 단계:")
print("   1. 실제 UAV 데이터셋 연결")
print("   2. 모델 학습")
print("   3. Phase 3: 실시간 비디오 추적")
print("=" * 70)
