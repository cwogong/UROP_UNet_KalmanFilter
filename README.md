# UROP_UNet_KalmanFilter

UAV 객체 추적을 위한 UNet과 Kalman Filter의 통합 프로젝트

## 프로젝트 구조

```
UROP_UNet_KalmanFilter/
├── combined_model/
│   └── unet_kalman_combined.py  # UNet + Kalman Filter 통합 모델
├── dataset/
│   └── uav_dataset.py           # UAV 데이터셋
├── filters/
│   └── linear_kalman_filter.py  # 2D 선형 Kalman Filter
├── model/
│   ├── UNet_center.py           # (미구현)
│   └── Vanilla_UNet.py          # Vanilla UNet 구현
├── experiments/
│   └── phase1_test.py           # Phase 1 테스트
└── config.yaml                  # 설정 파일
```

## Phase 2: 통합 (1-2주)

### 2.1 UNet + Kalman Filter 통합 아키텍처

- **프레임 단위 입력 처리**
  - UNet: 현재 프레임 → 객체 마스크 생성
  - Kalman Filter: 이전 프레임의 마스크 정보 → 현재 프레임 예측

- **데이터 흐름**
  ```
  입력 프레임
  ↓
  [UNet] → 객체 마스크
  ↓
  [Kalman Filter] → 평활화된 마스크
  ↓
  출력
  ```

### 2.2 시간 축 정보 활용

- 연속 프레임에서 마스크의 움직임 추적
- Kalman Filter로 노이즈 제거 및 예측

## 통합 모델 사용법

```python
from combined_model.unet_kalman_combined import UNetKalmanCombined
import torch
import numpy as np

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
    'x0': np.array([240, 240, 0, 0]),  # 초기 중심점
}

# 모델 생성
model = UNetKalmanCombined(unet_config, kalman_config)

# 단일 프레임 처리
frame = torch.randn(1, 3, 480, 480)
smoothed_mask = model(frame)

# 시퀀스 처리
frames = [torch.randn(1, 3, 480, 480) for _ in range(10)]
smoothed_masks = model.process_sequence(frames)
```

## 설치 및 실행

1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. 모델 테스트:
   ```bash
   python combined_model/unet_kalman_combined.py
   ```
