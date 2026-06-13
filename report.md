# Anti-UAV 환경에서 UNet 세그멘테이션과 칼만 필터를 융합한 드론 위치 추적 시스템

## 1. 서론

### 1.1 연구 배경

소형 무인 항공기(UAV, 드론)의 비인가 침입은 공항, 군사시설, 주요 인프라 등에 심각한 보안 위협이 되고 있다. 이에 대응하기 위한 Anti-UAV 시스템에서는 드론을 실시간으로 탐지하고 추적하는 기술이 핵심이다.

기존 방식은 객체 탐지 모델로 바운딩 박스를 추출한 뒤, 등속도 운동을 가정한 단순 추적기를 사용하는 구조가 일반적이다. 그러나 이 방식은 (1) 픽셀 수준의 정밀한 형태 정보를 활용하지 못하며, (2) 프레임 간 노이즈에 의한 추적 불안정(jitter)이 발생하고, (3) 비선형적인 비행 궤적에 대한 예측이 불가능하다는 한계를 갖는다.

### 1.2 연구 목적

본 연구는 UNet 기반 세그멘테이션과 칼만 필터(Kalman Filter)를 융합하여, 픽셀 단위의 드론 형태 분리와 시간축 정보를 활용한 안정적 추적을 동시에 달성하는 시스템을 구현한다. 구체적으로 다음을 목표로 한다:

1. UNet으로 프레임별 드론 세그멘테이션 마스크를 생성하고, 중심점(centroid)을 추출
2. 칼만 필터로 시간축 평활화를 적용하여 추적 안정성을 개선
3. 선형(CV), 등가속도(CA), 비선형(EKF-CTRV) 모션 모델을 비교 평가

### 1.3 연구 범위

- 데이터셋: ANTI-UAV RGBT (적외선/가시광 다중 모달리티)
- 세그멘테이션: VanillaUNet (Encoder-Decoder 구조)
- 추적 필터: Linear KF (CV), Constant Acceleration (CA), Extended KF (CTRV)
- 평가 지표: mIoU, Dice, CLE, Jitter, Smoothness Ratio

---

## 2. 관련 연구

### 2.1 UNet 기반 세그멘테이션

UNet은 Ronneberger et al. (2015)이 의료 영상 분할을 위해 제안한 Encoder-Decoder 구조로, skip connection을 통해 저수준 특징과 고수준 특징을 효과적으로 결합한다. 본 연구에서는 UAV 세그멘테이션에 적용하여 픽셀 단위의 마스크를 생성한다.

### 2.2 칼만 필터

칼만 필터(Kalman, 1960)는 노이즈가 포함된 관측값으로부터 시스템 상태를 최적 추정하는 재귀적 알고리즘이다. 객체 추적에서는 이전 위치로부터 다음 위치를 예측(predict)하고, 실제 관측값으로 보정(update)하는 과정을 반복한다.

- **선형 칼만 필터 (Linear KF)**: 상태 전이가 선형인 경우 적용. 등속도(Constant Velocity) 모델이 대표적.
- **확장 칼만 필터 (EKF)**: 비선형 상태 전이에 대해 야코비안으로 선형 근사하여 적용.

### 2.3 Anti-UAV 추적

Anti-UAV 연구에서는 주로 SiamFC, DiMP 등의 단일 객체 추적기(SOT)를 사용한다. 본 연구는 세그멘테이션 기반 접근으로, 마스크에서 중심점을 추출하여 칼만 필터의 측정값으로 활용하는 차별화된 파이프라인을 제안한다.

---

## 3. 제안 방법

### 3.1 전체 아키텍처

```
입력 프레임 (480×480×3)
       ↓
   [UNet Segmentation]
       ↓
  객체 마스크 (480×480×1)
       ↓
   [Centroid 추출] → 측정값 z = [x, y]
       ↓
   [Kalman Filter]
     - Predict: 이전 상태로부터 현재 위치 예측
     - Update: UNet 측정값으로 보정
       ↓
  평활화된 위치 추정
```

### 3.2 UNet 세그멘테이션

VanillaUNet 구조를 사용하며, 주요 구성은 다음과 같다:

| 파라미터 | 값 |
|----------|-----|
| 입력 채널 | 3 (RGB) |
| 시작 채널 | 32 |
| 깊이 (depth) | 4 |
| 출력 클래스 | 1 (이진 분할) |
| 활성 함수 | ReLU + BatchNorm |
| 손실 함수 | BCEWithLogitsLoss |
| Optimizer | AdamW (lr=1e-4) |

학습 후 마스크를 sigmoid로 변환하고, threshold(0.5)로 이진화하여 중심점을 추출한다.

### 3.3 모션 모델 비교

#### 3.3.1 Constant Velocity (CV) — 선형 칼만 필터

상태: **[x, y, vx, vy]** (4차원)

상태 전이:
```
x(k+1) = x(k) + vx·dt
y(k+1) = y(k) + vy·dt
vx(k+1) = vx(k)
vy(k+1) = vy(k)
```

가정: 속도가 일정. 가속/선회에 취약하나, 관측 대비 상태 차원이 적정하여 안정적.

#### 3.3.2 Constant Acceleration (CA)

상태: **[x, y, vx, vy, ax, ay]** (6차원)

상태 전이:
```
x(k+1)  = x(k)  + vx·dt + 0.5·ax·dt²
y(k+1)  = y(k)  + vy·dt + 0.5·ay·dt²
vx(k+1) = vx(k) + ax·dt
vy(k+1) = vy(k) + ay·dt
ax(k+1) = ax(k)
ay(k+1) = ay(k)
```

장점: 여전히 선형 모델이므로 야코비안 불필요. 가속/감속 구간에서 CV보다 정확.

#### 3.3.3 CTRV (Constant Turn Rate and Velocity) — EKF

상태: **[x, y, v, θ, ω]** (5차원, v=속력, θ=진행방향, ω=회전속도)

비선형 상태 전이:
```
x(k+1) = x(k) + v/ω · [sin(θ + ω·dt) - sin(θ)]    (ω≠0)
y(k+1) = y(k) + v/ω · [-cos(θ + ω·dt) + cos(θ)]   (ω≠0)
v(k+1) = v(k)
θ(k+1) = θ(k) + ω·dt
ω(k+1) = ω(k)
```

EKF는 야코비안 ∂f/∂x를 매 스텝 계산하여 공분산을 전파한다. 선회/곡선 운동에 적합하나, 관측값(2D)으로 5개 상태를 추정해야 하므로 불확실성이 크다.

### 3.4 시퀀스 전환 처리

다중 시퀀스 데이터에서 시퀀스가 바뀌면 객체 위치가 불연속적으로 점프한다. 이를 감지하여 칼만 필터를 리셋하는 로직을 적용했다:

```
if |center(t) - center(t-1)| > 100px:
    Kalman Filter 리셋 (현재 위치로 재초기화)
```

---

## 4. 실험

### 4.1 실험 환경

| 항목 | 내용 |
|------|------|
| 데이터셋 | ANTI-UAV RGBT (Masked_SAM2) |
| 학습 시퀀스 | 4 |
| 테스트 시퀀스 | 1 (1,708 프레임) |
| 이미지 크기 | 480 × 480 |
| 학습 Epoch | 50 |
| GPU | CUDA |
| 프레임워크 | PyTorch |

### 4.2 평가 지표

| 지표 | 설명 | 방향 |
|------|------|------|
| **mIoU** | 세그멘테이션 마스크의 교집합/합집합 비율 | ↑ |
| **Dice** | F1 기반 마스크 정확도 | ↑ |
| **CLE** | 예측 중심점과 GT 중심점의 유클리드 거리 (pixels) | ↓ |
| **Jitter** | 연속 프레임 간 가속도 변화량의 평균 (pixels/frame²) | ↓ |
| **Smoothness Ratio** | Raw jitter / Filtered jitter (>1이면 개선) | ↑ |

### 4.3 하이퍼파라미터 설정

Linear KF에서 Q(process noise)에 따른 Trade-off:

| Q | R | CLE (px) | Jitter 감소율 | 특성 |
|---|---|----------|--------------|------|
| 0.01 | 0.5 | +2.67 | 54% | 최대 평활화, 위치 lag 발생 |
| 0.1 | 0.5 | +0.79 | 34% | 균형 후보 |
| **0.3** | **0.5** | **+0.34** | **24%** | **채택 (CLE 손해 최소 + 유의미한 평활화)** |
| 1.0 | 0.5 | +0.10 | 14% | 위치 정확, 평활화 미미 |

Q가 작을수록 모델(등속도 가정)을 신뢰하여 평활화 효과가 크지만, 실제 위치 변화를 따라가지 못해 CLE가 증가한다. Q=0.3을 채택하였다.

---

## 5. 결과 및 분석

### 5.1 세그멘테이션 성능

| Model | mIoU | Dice |
|-------|------|------|
| VanillaUNet | 0.866 | 0.926 |

UNet 세그멘테이션 성능은 모든 필터 실험에서 동일하다 (칼만 필터는 마스크 자체를 변경하지 않음).

### 5.2 필터 비교 (Q=0.3, R=0.5)

| Method | CLE (px) ↓ | Jitter ↓ | Jitter 감소 | Smoothness ↑ |
|--------|-----------|----------|-------------|--------------|
| **Baseline (UNet only)** | **2.88** | 3.63 | — | 1.00 |
| Linear KF (CV) | 3.22 | **2.75** | **-24.0%** | **1.32** |
| Constant Acceleration (CA) | 3.13 | 3.08 | -15.0% | 1.18 |
| EKF (CTRV) | 3.27 | 3.35 | -7.5% | 1.08 |

### 5.3 결과 분석

**1) 선형 칼만 필터(CV)가 종합적으로 최적**

- Jitter 24% 감소, Smoothness 1.32로 추적 안정성이 크게 개선
- CLE 증가(+0.34px)는 전체 이미지 대비 무시할 수준 (480px 중 0.07%)

**2) CA 모델은 CLE에서 소폭 우위**

- CLE: 3.13px (CV: 3.22px) — 가속도 상태 추정으로 위치 lag 감소
- 그러나 Jitter 감소는 CV보다 열세 (15% vs 24%)

**3) EKF(CTRV)는 오히려 성능 저하**

- 5개 상태를 2개 관측값으로 추정하는 구조적 한계
- 직선 위주의 운동에서 heading/yaw_rate 추정이 불안정
- 급선회가 빈번한 시나리오에서만 이점이 발현될 것으로 예상

**4) 칼만 필터의 주된 기여는 "정확도"가 아닌 "안정성"**

- UNet 자체의 CLE가 이미 2.88px로 매우 정확
- 칼만 필터는 프레임 간 떨림(jitter)을 줄여 궤적의 연속성과 부드러움을 보장

### 5.4 Trade-off 분석

Q 파라미터에 따라 위치 정확도(CLE)와 추적 안정성(Jitter)은 상충 관계에 있다:

- Q ↓: 모델 신뢰도 증가 → Jitter 감소 but CLE 증가 (lag)
- Q ↑: 측정값 추종 → CLE 유지 but Jitter 감소 효과 감소

이 trade-off의 최적점은 응용 분야에 따라 결정된다. 실시간 추적 시스템에서는 안정성(Jitter)이, 정밀 위치 추정에서는 CLE가 우선될 수 있다.

---

## 6. 결론 및 향후 계획

### 6.1 결론

본 연구에서는 UNet 세그멘테이션과 칼만 필터를 융합한 Anti-UAV 드론 추적 시스템을 구현하고, 세 가지 모션 모델(CV, CA, EKF-CTRV)의 성능을 비교 평가하였다. 주요 결론은 다음과 같다:

1. **UNet 세그멘테이션**은 mIoU 0.866, Dice 0.926의 높은 정확도를 달성하였다.
2. **선형 칼만 필터(CV)**는 추적 안정성(Jitter 24% 감소)을 효과적으로 개선하며, 위치 정확도 손실이 최소(+0.34px)인 최적의 필터였다.
3. **등가속도(CA) 모델**은 위치 정확도에서 소폭 우위를 보였으나, 안정성 개선은 제한적이었다.
4. **EKF(CTRV)**는 상태 차원과 관측 차원의 불균형으로 인해 본 데이터셋에서는 효과가 미미하였다.
5. 칼만 필터의 핵심 기여는 위치 정확도 향상보다 **추적 궤적의 시간적 안정성 확보**에 있다.

### 6.2 한계점

- 단일 객체 추적만 지원 (다중 드론 시나리오 미대응)
- 테스트 시퀀스가 1개로, 다양한 운동 패턴에 대한 일반화 검증 부족
- EKF의 잠재력을 확인하기 위한 급선회/고기동 시퀀스 부재
- 가림(occlusion) 상황에서의 추적 연속성 미검증

### 6.3 향후 연구

1. **다양한 운동 시나리오 확보**: 급선회, 호버링, 급가속 시퀀스에서 EKF 재평가
2. **적응형 Q/R 추정**: IMM(Interacting Multiple Model) 필터로 모션 모드 자동 전환
3. **가림 대응**: 검출 실패 시 칼만 필터의 예측만으로 추적 유지하는 전략 검증
4. **실시간 처리**: 추론 속도 최적화 및 실시간 시스템 통합

---

## 참고문헌

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. ASME Journal of Basic Engineering.
3. Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter. UNC Chapel Hill TR 95-041.

---

## 부록: 프로젝트 구조 및 실행 방법

### 프로젝트 구조
```
UROP_UNet_KalmanFilter/
├── model/Vanilla_UNet.py              # UNet 구현
├── filters/
│   ├── linear_kalman_filter.py        # 선형 KF (CV)
│   ├── constant_acceleration_filter.py # CA 필터
│   └── extended_kalman_filter.py      # EKF (CTRV)
├── combined_model/
│   └── unet_kalman_combined.py        # 통합 모델
├── dataset/uav_dataset.py             # 데이터 로딩
├── trainer.py                         # 학습 (진행률 시각화 포함)
├── tester.py                          # 평가 (sweep, 필터 비교)
├── eval/
│   ├── metrics.py                     # 평가 지표
│   ├── evaluate.py                    # 합성 데이터 평가
│   └── noise_sensitivity.py           # 노이즈 민감도 분석
└── config.yaml                        # 설정 파일
```

### 실행 명령어
```bash
# 학습
python trainer.py --config config.yaml --epochs 50

# 평가 (4자 비교)
python tester.py --checkpoint checkpoints/demo_unet.pth --filter all --save-vis

# Q 파라미터 sweep
python tester.py --checkpoint checkpoints/demo_unet.pth --sweep --q-values "0.01,0.05,0.1,0.3,0.5,1.0"
```
