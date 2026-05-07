#!/usr/bin/env python3
"""
간단한 테스트: Phase 2 모델 로드 확인
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Python 버전:", sys.version)
print("프로젝트 경로:", project_root)

try:
    print("\n1. 라이브러리 임포트 확인...")
    import torch
    print("   ✓ torch 버전:", torch.__version__)
    
    import numpy as np
    print("   ✓ numpy 버전:", np.__version__)
    
    import cv2
    print("   ✓ OpenCV 버전:", cv2.__version__)
    
    from scipy import ndimage
    print("   ✓ scipy 임포트 성공")
    
    print("\n2. 모델 임포트 확인...")
    from model.Vanilla_UNet import VanillaUNet
    print("   ✓ VanillaUNet 임포트 성공")
    
    from filters.linear_kalman_filter import KalmanFilter
    print("   ✓ KalmanFilter 임포트 성공")
    
    print("\n3. 통합 모델 임포트 확인...")
    from combined_model.unet_kalman_combined import UNetKalmanCombined
    print("   ✓ UNetKalmanCombined 임포트 성공")
    
    print("\n4. 유틸리티 임포트 확인...")
    from combined_model.mask_utils import MaskProcessor, TrackingVisualizer
    print("   ✓ MaskProcessor 임포트 성공")
    print("   ✓ TrackingVisualizer 임포트 성공")
    
    print("\n✅ 모든 임포트 성공!")
    print("\n다음 단계: python experiments/phase2_test.py 실행")
    
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
