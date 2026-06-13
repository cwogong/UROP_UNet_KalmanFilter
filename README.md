# Anti-UAV Tracking: UNet + Kalman Filter

UAV 객체 추적을 위한 UNet 세그멘테이션과 칼만 필터 통합 시스템.

## 프로젝트 구조

```
UROP_UNet_KalmanFilter/
├── model/
│   ├── Vanilla_UNet.py                # UNet 세그멘테이션 네트워크
│   └── UNet_center.py                 # UNet + 중심점 추출
├── filters/
│   ├── linear_kalman_filter.py        # 선형 KF (Constant Velocity)
│   ├── constant_acceleration_filter.py # CA 필터 (Constant Acceleration)
│   └── extended_kalman_filter.py      # EKF (CTRV 모션 모델)
├── combined_model/
│   ├── unet_kalman_combined.py        # UNet + Kalman 통합 파이프라인
│   └── mask_utils.py                  # 마스크 처리/시각화 유틸리티
├── dataset/
│   └── uav_dataset.py                 # ANTI-UAV 데이터셋 로더
├── eval/
│   ├── metrics.py                     # 평가 지표 (IoU, Dice, CLE, Jitter)
│   ├── evaluate.py                    # 합성 데이터 평가 + Q/R 튜닝
│   ├── noise_sensitivity.py           # 노이즈 민감도 분석
│   └── run_phase3.py                  # Phase 3 전체 파이프라인
├── experiments/
│   ├── phase1_test.py                 # Kalman Filter 단독 테스트
│   ├── phase2_test.py                 # UNet+Kalman 통합 테스트
│   └── eval_checkpoint.py             # 체크포인트 평가
├── trainer.py                         # 학습 (tqdm + 실시간 커브 시각화)
├── tester.py                          # 테스트 (필터 비교, Q sweep)
├── config.yaml                        # 기본 설정 (channels=32, depth=4)
├── config_light.yaml                  # 경량 설정 (channels=16, depth=3)
└── requirements.txt                   # 의존성
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
# 기본 UNet 학습
python trainer.py --config config.yaml --epochs 50

# 경량 UNet 학습
python trainer.py --config config_light.yaml --epochs 50 --save-dir checkpoints_light
```

### 테스트 (필터 비교)

```bash
# Baseline vs Linear KF vs CA vs EKF 전체 비교
python tester.py --checkpoint checkpoints/demo_unet.pth --filter all --save-vis

# Linear KF만 테스트
python tester.py --checkpoint checkpoints/demo_unet.pth --filter linear

# Q 파라미터 sweep (다중 Q 한번에 테스트)
python tester.py --checkpoint checkpoints/demo_unet.pth --sweep --q-values "0.01,0.05,0.1,0.3,0.5,1.0"
```

### 합성 데이터 평가 (GPU 불필요)

```bash
python eval/run_phase3.py
python eval/noise_sensitivity.py
```

## 핵심 결과

| 방법 | CLE (px) ↓ | Jitter ↓ | Jitter 감소 | Smoothness ↑ |
|------|-----------|----------|-------------|--------------|
| Baseline (UNet only) | 2.88 | 3.63 | — | 1.00 |
| **Linear KF (CV)** | 3.22 | **2.75** | **-24%** | **1.32** |
| CA (등가속도) | 3.13 | 3.08 | -15% | 1.18 |
| EKF (CTRV) | 3.27 | 3.35 | -7.5% | 1.08 |

- 선형 칼만 필터(등속도 모델)가 종합 최적
- 주된 기여: 추적 안정성(Jitter 감소), 위치 정확도 손실 최소

## 설정 (config.yaml)

```yaml
model:
  name: VanillaUNet
  in_channels: 3
  start_out_channels: 32
  num_class: 1
  size: 4
  padding: 1

kalman:
  type: linear          # 'linear', 'ekf'
  dt: 1.0
  process_noise: 0.3    # Q (작을수록 평활화↑, CLE↑)
  measurement_noise: 0.5 # R
```

## 연구 단계

- [x] Phase 1: 선형 Kalman Filter 구현 및 검증
- [x] Phase 2: UNet + Kalman Filter 통합
- [x] Phase 3: 성능 평가 (Baseline vs Kalman, Q/R 튜닝)
- [x] Phase 4: EKF(CTRV), CA(등가속도) 모델 비교
