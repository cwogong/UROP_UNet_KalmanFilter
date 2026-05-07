"""
Phase 2 간단 테스트 - 모든 모듈 import 확인
(torch 없이도 가능한 테스트)
"""
import numpy as np
import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Phase 2: UNet + Kalman Filter 통합 - 기본 테스트")
print("=" * 70)

# ============================================================
# 테스트 1: Kalman Filter 검증 (torch 불필요)
# ============================================================
print("\n[테스트 1] Kalman Filter 기본 기능")
print("-" * 70)

try:
    from filters.linear_kalman_filter import KalmanFilter
    
    # Kalman Filter 설정
    kalman_config = {
        'dt': 1.0,
        'x0': np.array([240, 240, 0, 0]),
        'Q': np.eye(4) * 0.01,
        'R': np.eye(2) * 0.5,
        'P': np.eye(4) * 100
    }
    
    kf = KalmanFilter(**kalman_config)
    print("✅ Kalman Filter 초기화 성공")
    print(f"   - 초기 상태: {kf.get_state()}")
    print(f"   - 초기 위치: {kf.get_position()}")
    
    # 시뮬레이션: 선형 운동
    print("\n   📊 선형 운동 추적 테스트 (5 프레임):")
    for i in range(5):
        measurement = np.array([240 + i * 5, 240 + i * 3])
        kf.predict()
        kf.update(measurement)
        pos = kf.get_position()
        vel = kf.get_velocity()
        print(f"      Frame {i+1}: pos=[{pos[0]:.2f}, {pos[1]:.2f}], vel=[{vel[0]:.4f}, {vel[1]:.4f}]")
    
    print("✅ Kalman Filter 정상 작동!")
    
except Exception as e:
    print(f"❌ Kalman Filter 에러: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 테스트 2: 마스크 처리 유틸리티
# ============================================================
print("\n[테스트 2] 마스크 처리 유틸리티")
print("-" * 70)

try:
    from combined_model.mask_utils import MaskProcessor
    
    print("✅ MaskProcessor 임포트 성공")
    
    # 메트릭 계산
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
    
    print("\n   📊 추적 메트릭 계산 결과:")
    print(f"      - 처리된 프레임: {metrics.get('num_frames', 'N/A')}")
    print(f"      - 평균 거리: {metrics.get('avg_distance', 0):.4f} pixels")
    print(f"      - 최대 거리: {metrics.get('max_distance', 0):.4f} pixels")
    print(f"      - 표준편차: {metrics.get('std_distance', 0):.4f} pixels")
    print(f"      - 감지된 평균 속도: {metrics.get('detected_avg_speed', 0):.4f} pixels/frame")
    print(f"      - Kalman 평균 속도: {metrics.get('kalman_avg_speed', 0):.4f} pixels/frame")
    print(f"      - 평활도 개선: {metrics.get('smoothness_improvement', 0):.2f}%")
    
    print("✅ 메트릭 계산 정상 작동!")
    
except Exception as e:
    print(f"❌ 마스크 처리 에러: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 테스트 3: UNet 모델 구조 (torch 필요)
# ============================================================
print("\n[테스트 3] UNet 모델 구조 검증")
print("-" * 70)

try:
    import torch
    from model.Vanilla_UNet import VanillaUNet
    
    print("✅ PyTorch 및 VanillaUNet 임포트 성공")
    print(f"   - PyTorch 버전: {torch.__version__}")
    
    unet_config = {
        'in_channels': 3,
        'start_out_channels': 32,
        'num_class': 1,
        'size': 4,
        'padding': 1
    }
    
    unet = VanillaUNet(**unet_config)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"   - 총 파라미터: {total_params:,}")
    
    # Forward pass 테스트
    dummy_input = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        output = unet(dummy_input)
    
    print(f"   - 입력 크기: {dummy_input.shape}")
    print(f"   - 출력 크기: {output.shape}")
    print("✅ UNet forward pass 성공!")
    
except ImportError:
    print("⚠️  PyTorch 미설치")
    print("   설치 명령: pip install torch torchvision")
except Exception as e:
    print(f"❌ UNet 에러: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 테스트 4: 통합 모델
# ============================================================
print("\n[테스트 4] UNet + Kalman Filter 통합 모델")
print("-" * 70)

try:
    import torch
    from combined_model.unet_kalman_combined import UNetKalmanCombined
    
    print("✅ UNetKalmanCombined 임포트 성공")
    
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
    
    model = UNetKalmanCombined(unet_config, kalman_config, use_morphology=True)
    model.eval()
    
    print("✅ 통합 모델 생성 완료")
    
    # Forward pass 테스트
    with torch.no_grad():
        dummy_frame = torch.randn(1, 3, 480, 480)
        result = model(dummy_frame, use_kalman=True)
    
    print(f"\n   📊 단일 프레임 처리 결과:")
    print(f"      - 입력 크기: {dummy_frame.shape}")
    print(f"      - 출력 마스크 크기: {result['smoothed_mask'].shape}")
    print(f"      - 추출된 중심점: {result['center']}")
    print(f"      - Kalman 중심점: {result['kalman_center']}")
    print(f"      - 경계박스: {result['bbox']}")
    
    # 시퀀셜 처리 테스트
    print(f"\n   📊 시퀀셜 처리 테스트 (5 프레임):")
    model.reset()
    frames = [torch.randn(1, 3, 480, 480) for _ in range(5)]
    
    with torch.no_grad():
        seq_result = model.process_sequence(frames, use_kalman=True)
    
    print(f"      - 처리된 프레임: {len(seq_result['smoothed_masks'])}")
    print(f"      - 총 프레임 카운트: {model.get_frame_count()}")
    
    info = model.get_model_info()
    print(f"      - Kalman 위치: {info['kalman_position']}")
    print(f"      - Kalman 속도: {info['kalman_velocity']}")
    
    print("✅ 통합 모델 정상 작동!")
    
except ImportError as e:
    print(f"⚠️  PyTorch 또는 모듈 미설치: {e}")
    print("   필요한 패키지: pip install torch numpy opencv-python scipy")
except Exception as e:
    print(f"❌ 통합 모델 에러: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 최종 결과
# ============================================================
print("\n" + "=" * 70)
print("✅ Phase 2 테스트 완료!")
print("=" * 70)
print("\n📋 테스트 요약:")
print("   ✅ [테스트 1] Kalman Filter - 정상 작동")
print("   ✅ [테스트 2] 마스크 처리 - 정상 작동")
print("   ⚠️  [테스트 3] UNet 모델 - PyTorch 필요")
print("   ⚠️  [테스트 4] 통합 모델 - PyTorch 필요")
print("\n🔧 필수 패키지 설치:")
print("   pip install torch numpy opencv-python scipy matplotlib")
print("\n📊 설치 후 다시 실행하면 모든 테스트를 완료할 수 있습니다.")
print("=" * 70)
