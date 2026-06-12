"""
Phase 3: 평가 지표 모듈

지표:
1. mIoU (Mean Intersection over Union)
2. Dice Score (F1)
3. 추적 안정성 (Tracking Jitter) — 연속 프레임 간 중심점 변동
4. 추적 연속성 (Tracking Continuity) — 객체 검출 유지 비율
5. 위치 정확도 (CLE: Center Location Error)
"""

import numpy as np
from typing import List, Optional, Dict


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    IoU (Intersection over Union) 계산

    Args:
        pred_mask: 예측 마스크 (H, W), 0~1 범위
        gt_mask: GT 마스크 (H, W), 0~1 범위
        threshold: 이진화 임계값

    Returns:
        float: IoU 값 (0~1)
    """
    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > threshold).astype(np.float32)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

    if union == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0

    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Dice Score (F1) 계산

    Args:
        pred_mask: 예측 마스크 (H, W), 0~1 범위
        gt_mask: GT 마스크 (H, W), 0~1 범위
        threshold: 이진화 임계값

    Returns:
        float: Dice score (0~1)
    """
    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > threshold).astype(np.float32)

    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)

    if total == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0

    return float(2.0 * intersection / total)


def compute_center_location_error(
    pred_center: Optional[np.ndarray],
    gt_center: Optional[np.ndarray]
) -> Optional[float]:
    """
    CLE (Center Location Error) — 유클리드 거리

    Args:
        pred_center: 예측 중심점 [x, y]
        gt_center: GT 중심점 [x, y]

    Returns:
        float or None: 거리 (pixels)
    """
    if pred_center is None or gt_center is None:
        return None

    return float(np.linalg.norm(np.array(pred_center) - np.array(gt_center)))


def compute_tracking_jitter(centers: List[Optional[np.ndarray]]) -> Dict[str, float]:
    """
    추적 안정성 (Jitter) 측정

    연속 프레임 간 중심점 이동 속도의 변동성으로 측정.
    Jitter가 낮을수록 추적이 안정적.

    측정 방법:
    - velocity = center[t] - center[t-1]
    - jitter = std(velocity) — 속도의 표준편차

    Args:
        centers: 프레임별 중심점 리스트 [np.array([x,y]) or None, ...]

    Returns:
        dict: {
            'mean_jitter': 평균 jitter (pixels/frame),
            'max_jitter': 최대 jitter,
            'velocity_std_x': x축 속도 표준편차,
            'velocity_std_y': y축 속도 표준편차,
        }
    """
    # None이 아닌 연속 쌍 찾기
    velocities = []
    for i in range(1, len(centers)):
        if centers[i] is not None and centers[i - 1] is not None:
            vel = np.array(centers[i]) - np.array(centers[i - 1])
            velocities.append(vel)

    if len(velocities) < 2:
        return {
            'mean_jitter': 0.0,
            'max_jitter': 0.0,
            'velocity_std_x': 0.0,
            'velocity_std_y': 0.0,
        }

    velocities = np.array(velocities)  # (N, 2)

    # Jitter = 속도 변화량의 표준편차 (acceleration의 크기)
    accelerations = np.diff(velocities, axis=0)  # (N-1, 2)
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    return {
        'mean_jitter': float(np.mean(accel_magnitudes)),
        'max_jitter': float(np.max(accel_magnitudes)),
        'velocity_std_x': float(np.std(velocities[:, 0])),
        'velocity_std_y': float(np.std(velocities[:, 1])),
    }


def compute_tracking_continuity(centers: List[Optional[np.ndarray]]) -> Dict[str, float]:
    """
    추적 연속성 측정

    객체가 검출되지 않은 프레임(center=None)의 비율로 판단.

    Args:
        centers: 프레임별 중심점 리스트

    Returns:
        dict: {
            'detection_rate': 검출 비율 (0~1),
            'max_gap': 최대 연속 미검출 프레임 수,
            'num_gaps': 미검출 구간 수,
        }
    """
    total = len(centers)
    if total == 0:
        return {'detection_rate': 0.0, 'max_gap': 0, 'num_gaps': 0}

    detected = sum(1 for c in centers if c is not None)
    detection_rate = detected / total

    # 최대 연속 미검출 구간
    max_gap = 0
    current_gap = 0
    num_gaps = 0

    for c in centers:
        if c is None:
            current_gap += 1
        else:
            if current_gap > 0:
                num_gaps += 1
                max_gap = max(max_gap, current_gap)
            current_gap = 0

    # 마지막 구간 처리
    if current_gap > 0:
        num_gaps += 1
        max_gap = max(max_gap, current_gap)

    return {
        'detection_rate': float(detection_rate),
        'max_gap': int(max_gap),
        'num_gaps': int(num_gaps),
    }


def compute_smoothness_ratio(
    raw_centers: List[Optional[np.ndarray]],
    filtered_centers: List[Optional[np.ndarray]]
) -> float:
    """
    평활도 비율 (Smoothness Ratio)

    Kalman 필터 적용 전/후의 jitter 비교.
    값이 1보다 크면 필터가 안정화에 기여.

    Returns:
        float: raw_jitter / filtered_jitter (>1이면 개선됨)
    """
    raw_jitter = compute_tracking_jitter(raw_centers)['mean_jitter']
    filtered_jitter = compute_tracking_jitter(filtered_centers)['mean_jitter']

    if filtered_jitter < 1e-8:
        return float('inf') if raw_jitter > 1e-8 else 1.0

    return float(raw_jitter / filtered_jitter)


class SequenceMetrics:
    """
    시퀀스 단위 평가 지표 수집기

    사용법:
        metrics = SequenceMetrics()
        for frame_idx in range(num_frames):
            metrics.update(pred_mask, gt_mask, pred_center, gt_center, kalman_center)
        result = metrics.summarize()
    """

    def __init__(self):
        self.ious = []
        self.dices = []
        self.cles_raw = []
        self.cles_kalman = []
        self.raw_centers = []
        self.kalman_centers = []
        self.gt_centers = []

    def update(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        raw_center: Optional[np.ndarray] = None,
        gt_center: Optional[np.ndarray] = None,
        kalman_center: Optional[np.ndarray] = None,
    ):
        """프레임 단위 업데이트"""
        # 마스크 지표
        self.ious.append(compute_iou(pred_mask, gt_mask))
        self.dices.append(compute_dice(pred_mask, gt_mask))

        # 중심점 추적
        self.raw_centers.append(raw_center)
        self.kalman_centers.append(kalman_center)
        self.gt_centers.append(gt_center)

        # CLE
        cle_raw = compute_center_location_error(raw_center, gt_center)
        cle_kalman = compute_center_location_error(kalman_center, gt_center)
        if cle_raw is not None:
            self.cles_raw.append(cle_raw)
        if cle_kalman is not None:
            self.cles_kalman.append(cle_kalman)

    def summarize(self) -> Dict[str, float]:
        """전체 시퀀스 요약 통계"""
        result = {}

        # 마스크 지표
        result['mIoU'] = float(np.mean(self.ious)) if self.ious else 0.0
        result['mean_dice'] = float(np.mean(self.dices)) if self.dices else 0.0

        # CLE (위치 정확도)
        result['mean_CLE_raw'] = float(np.mean(self.cles_raw)) if self.cles_raw else 0.0
        result['mean_CLE_kalman'] = float(np.mean(self.cles_kalman)) if self.cles_kalman else 0.0
        result['CLE_improvement'] = (
            result['mean_CLE_raw'] - result['mean_CLE_kalman']
        ) if self.cles_raw and self.cles_kalman else 0.0

        # 추적 안정성
        raw_jitter = compute_tracking_jitter(self.raw_centers)
        kalman_jitter = compute_tracking_jitter(self.kalman_centers)
        result['jitter_raw'] = raw_jitter['mean_jitter']
        result['jitter_kalman'] = kalman_jitter['mean_jitter']
        result['jitter_reduction'] = (
            (raw_jitter['mean_jitter'] - kalman_jitter['mean_jitter'])
            / (raw_jitter['mean_jitter'] + 1e-8) * 100
        )

        # 추적 연속성
        continuity_raw = compute_tracking_continuity(self.raw_centers)
        continuity_kalman = compute_tracking_continuity(self.kalman_centers)
        result['detection_rate_raw'] = continuity_raw['detection_rate']
        result['detection_rate_kalman'] = continuity_kalman['detection_rate']
        result['max_gap_raw'] = continuity_raw['max_gap']

        # 평활도 비율
        result['smoothness_ratio'] = compute_smoothness_ratio(
            self.raw_centers, self.kalman_centers
        )

        return result


if __name__ == '__main__':
    # 간단한 테스트
    print("=" * 60)
    print("Phase 3: 평가 지표 모듈 테스트")
    print("=" * 60)

    # 합성 데이터
    np.random.seed(42)
    H, W = 128, 128

    # GT: 원형 객체가 직선 이동
    metrics = SequenceMetrics()
    for t in range(30):
        gt_cx, gt_cy = 50 + t * 2, 64
        gt_mask = np.zeros((H, W))
        y, x = np.ogrid[:H, :W]
        gt_mask[((x - gt_cx) ** 2 + (y - gt_cy) ** 2) <= 15 ** 2] = 1.0

        # pred: GT + noise
        noise_cx = gt_cx + np.random.randn() * 3
        noise_cy = gt_cy + np.random.randn() * 3
        pred_mask = np.zeros((H, W))
        pred_mask[((x - noise_cx) ** 2 + (y - noise_cy) ** 2) <= 15 ** 2] = 1.0

        raw_center = np.array([noise_cx, noise_cy])
        kalman_center = np.array([gt_cx + np.random.randn() * 1, gt_cy + np.random.randn() * 1])
        gt_center = np.array([gt_cx, gt_cy])

        metrics.update(pred_mask, gt_mask, raw_center, gt_center, kalman_center)

    result = metrics.summarize()
    print("\n📊 평가 결과:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
