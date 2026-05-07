import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from cv2 import morphologyEx, MORPH_OPEN, MORPH_CLOSE, getStructuringElement, MORPH_ELLIPSE
from model.Vanilla_UNet import VanillaUNet
from filters.linear_kalman_filter import KalmanFilter

class UNetKalmanCombined(nn.Module):
    """
    통합 모델: UNet + Kalman Filter
    
    아키텍처:
    1. UNet: 입력 프레임 → 객체 세그멘테이션 마스크
    2. 마스크 후처리: 노이즈 제거, 이진화
    3. 객체 추적: 중심점/경계박스 추출
    4. Kalman Filter: 위치 평활화 및 예측
    
    지원 기능:
    - 단일/다중 프레임 처리
    - 시간축 정보 활용
    - 다중 객체 추적 (구현 가능)
    """

    def __init__(self, unet_config, kalman_config, mask_threshold=0.5, 
                 use_morphology=True, morphology_kernel_size=5):
        """
        Args:
            unet_config (dict): UNet 모델 설정
                - in_channels: 입력 채널 수
                - start_out_channels: 시작 출력 채널
                - num_class: 클래스 수
                - size: UNet 깊이
                - padding: 패딩
            kalman_config (dict): Kalman Filter 설정
                - dt: 시간 간격
                - x0: 초기 상태
                - Q: 프로세스 노이즈
                - R: 측정 노이즈
                - P: 초기 공분산
            mask_threshold (float): 마스크 이진화 임계값 (0~1)
            use_morphology (bool): 모폴로지 연산 사용 여부
            morphology_kernel_size (int): 모폴로지 연산 커널 크기
        """
        super(UNetKalmanCombined, self).__init__()
        self.unet = VanillaUNet(**unet_config)
        self.kalman = KalmanFilter(**kalman_config)
        
        # 마스크 처리 파라미터
        self.mask_threshold = mask_threshold
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        
        # 상태 추적
        self.prev_mask = None
        self.prev_center = None
        self.frame_count = 0

    def forward(self, frame, use_kalman=True):
        """
        프레임 단위 입력 처리 및 추적
        
        데이터 흐름:
        Frame → [UNet] → Raw Mask 
                          ↓
                      [Preprocessing] → Clean Mask
                          ↓
                      [Extract Center] → Position Measurement
                          ↓
                      [Kalman Update] → Filtered Position
                          ↓
                      [Smooth Mask] → Output

        Args:
            frame (torch.Tensor): 입력 프레임 (B, C, H, W), B=1 권장
            use_kalman (bool): Kalman Filter 사용 여부

        Returns:
            dict: 처리 결과
                - 'smoothed_mask': 평활화된 마스크 (B, 1, H, W)
                - 'raw_mask': 원본 마스크 (B, 1, H, W)
                - 'center': 추출된 중심점 [x, y]
                - 'kalman_center': Kalman 필터 예측 중심점 [x, y]
                - 'filtered_center': 최종 추적 중심점 [x, y]
                - 'bbox': 경계박스 (x1, y1, x2, y2)
        """
        self.frame_count += 1
        
        # 1. UNet으로 마스크 생성
        raw_mask = self.unet(frame)  # (B, C, H, W)
        raw_mask = torch.sigmoid(raw_mask)  # 확률로 변환
        
        # 배치 크기 1 권장
        B = frame.size(0)
        
        results = []
        
        for b in range(B):
            mask_np = raw_mask[b, 0].detach().cpu().numpy()  # (H, W)
            
            # 2. 마스크 전처리 (노이즈 제거)
            clean_mask = self._preprocess_mask(mask_np)
            
            # 3. 객체 중심점 추출
            center = self._extract_center(clean_mask)
            bbox = self._extract_bbox(clean_mask)
            
            # 4. Kalman Filter 업데이트
            if use_kalman:
                if center is not None:
                    self.kalman.predict()  # 예측
                    self.kalman.update(center)  # 업데이트
                    kalman_center = self.kalman.get_position()
                else:
                    self.kalman.predict()  # 측정 없음: 예측만
                    kalman_center = self.kalman.get_position()
            else:
                kalman_center = center if center is not None else np.array([0, 0])
            
            # 5. 마스크 평활화
            if center is not None and kalman_center is not None:
                smoothed_mask = self._smooth_mask(clean_mask, kalman_center)
            else:
                smoothed_mask = clean_mask
            
            # 결과 저장
            results.append({
                'smoothed_mask': torch.from_numpy(smoothed_mask).unsqueeze(0).float(),
                'raw_mask': raw_mask[b:b+1],
                'center': center,
                'kalman_center': kalman_center,
                'filtered_center': kalman_center if center is not None else None,
                'bbox': bbox
            })
        
        # 배치 처리
        if B == 1:
            result = results[0]
            result['smoothed_mask'] = result['smoothed_mask'].unsqueeze(0)
            result['raw_mask'] = result['raw_mask']
            self.prev_mask = result['smoothed_mask'].squeeze().detach().cpu().numpy()
            self.prev_center = result['filtered_center']
            return result
        else:
            # 배치로 처리할 경우
            smoothed_masks = torch.cat([r['smoothed_mask'] for r in results], dim=0).unsqueeze(1)
            return {
                'smoothed_mask': smoothed_masks,
                'raw_mask': raw_mask,
                'results': results
            }

    def _preprocess_mask(self, mask):
        """
        마스크 전처리: 노이즈 제거 및 이진화
        
        처리 과정:
        1. 이진화 (임계값)
        2. 모폴로지 연산 (선택사항): Opening (잡음 제거) → Closing (구멍 메우기)
        3. 가우시안 필터 (부드럽게)

        Args:
            mask (np.ndarray): 마스크 (H, W), 0-1 범위

        Returns:
            np.ndarray: 전처리된 마스크 (H, W)
        """
        # 1. 이진화
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)
        
        # 2. 모폴로지 연산 (선택사항)
        if self.use_morphology:
            kernel = getStructuringElement(MORPH_ELLIPSE, 
                                          (self.morphology_kernel_size, 
                                           self.morphology_kernel_size))
            # Opening: 작은 잡음 제거
            binary_mask = morphologyEx(binary_mask, MORPH_OPEN, kernel, iterations=1)
            # Closing: 작은 구멍 메우기
            binary_mask = morphologyEx(binary_mask, MORPH_CLOSE, kernel, iterations=1)
        
        # 3. 작은 연결 성분 제거 (객체 크기 필터링)
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
            min_size = 10  # 최소 픽셀 수
            mask_sizes = sizes < min_size
            binary_mask[mask_sizes[labeled]] = 0
        
        return binary_mask.astype(np.float32)

    def _extract_center(self, mask):
        """
        마스크에서 객체의 중심점 추출 (무게 중심)

        Args:
            mask (np.ndarray): 마스크 (H, W), 0-1 또는 0-255 범위

        Returns:
            np.ndarray or None: 중심점 [x, y] or None (객체 없음)
        """
        binary_mask = (mask > 0.5).astype(np.uint8) if mask.max() <= 1 else (mask > 127).astype(np.uint8)

        # 객체가 있는지 확인
        if np.sum(binary_mask) == 0:
            return None

        # 무게 중심 계산
        indices = np.argwhere(binary_mask > 0)
        center_y, center_x = np.mean(indices, axis=0)

        return np.array([center_x, center_y], dtype=np.float32)

    def _extract_bbox(self, mask):
        """
        마스크에서 객체의 경계박스 추출

        Args:
            mask (np.ndarray): 마스크 (H, W)

        Returns:
            tuple or None: (x1, y1, x2, y2) or None (객체 없음)
        """
        binary_mask = (mask > 0.5).astype(np.uint8) if mask.max() <= 1 else (mask > 127).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            return None

        indices = np.argwhere(binary_mask > 0)
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)

        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def _smooth_mask(self, mask, predicted_center):
        """
        Kalman 필터 예측 위치를 사용해 마스크 평활화
        
        전략: 
        1. 원본 마스크의 중심점 계산
        2. 예측 중심점과의 오프셋 계산
        3. 마스크를 오프셋만큼 이동

        Args:
            mask (np.ndarray): 원본 마스크 (H, W)
            predicted_center (np.ndarray): 예측 중심점 [x, y]

        Returns:
            np.ndarray: 평활화된 마스크 (H, W)
        """
        original_center = self._extract_center(mask)
        if original_center is None:
            return mask

        # 이동 벡터 계산
        shift = predicted_center - original_center
        shift_x, shift_y = shift.astype(np.float32)

        # 마스크 이동 (순환 교대 좌표)
        h, w = mask.shape
        from scipy.ndimage import shift as scipy_shift
        
        try:
            smoothed_mask = scipy_shift(mask, shift=(shift_y, shift_x), 
                                       order=1, mode='constant', cval=0.0)
        except Exception as e:
            print(f"Shift warning: {e}")
            smoothed_mask = mask

        return smoothed_mask

    def process_sequence(self, frames, use_kalman=True):
        """
        연속 프레임 시퀀스 처리 (시간 축 정보 활용)
        
        각 프레임을 순차적으로 처리하며 Kalman Filter는 상태를 유지

        Args:
            frames (list of torch.Tensor): 프레임 리스트 [(1, C, H, W), ...]
            use_kalman (bool): Kalman Filter 사용 여부

        Returns:
            dict: 처리 결과
                - 'smoothed_masks': 평활화된 마스크 리스트
                - 'raw_masks': 원본 마스크 리스트
                - 'centers': 중심점 리스트
                - 'kalman_centers': Kalman 예측 중심점 리스트
                - 'bboxes': 경계박스 리스트
        """
        smoothed_masks = []
        raw_masks = []
        centers = []
        kalman_centers = []
        bboxes = []
        
        for frame_idx, frame in enumerate(frames):
            result = self.forward(frame, use_kalman=use_kalman)
            
            smoothed_masks.append(result['smoothed_mask'])
            raw_masks.append(result['raw_mask'])
            centers.append(result['center'])
            kalman_centers.append(result['kalman_center'])
            bboxes.append(result['bbox'])
        
        return {
            'smoothed_masks': smoothed_masks,
            'raw_masks': raw_masks,
            'centers': centers,
            'kalman_centers': kalman_centers,
            'bboxes': bboxes
        }

    def reset(self):
        """전체 상태 초기화"""
        self.kalman.reset()
        self.prev_mask = None
        self.prev_center = None
        self.frame_count = 0

    def reset_kalman(self):
        """Kalman Filter만 리셋"""
        self.kalman.reset()

    def get_kalman_state(self):
        """Kalman Filter 현재 상태 반환"""
        return self.kalman.get_state()

    def set_kalman_state(self, state):
        """Kalman Filter 상태 설정"""
        self.kalman.set_state(state)

    def get_frame_count(self):
        """처리한 프레임 수 반환"""
        return self.frame_count

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'frame_count': self.frame_count,
            'kalman_state': self.kalman.get_state(),
            'kalman_position': self.kalman.get_position(),
            'kalman_velocity': self.kalman.get_velocity()
        }

# 사용 예시
if __name__ == '__main__':
    # 설정
    unet_config = {
        'in_channels': 3,
        'start_out_channels': 32,
        'num_class': 1,
        'size': 4,
        'padding': 1
    }

    kalman_config = {
        'dt': 1.0,  # 프레임 간 시간 간격
        'x0': np.array([240, 240, 0, 0]),  # 초기 중심점
        'Q': np.eye(4) * 0.01,
        'R': np.eye(2) * 0.5,
        'P': np.eye(4) * 100
    }

    print("=" * 60)
    print("Phase 2: UNet + Kalman Filter 통합 모델 테스트")
    print("=" * 60)

    # 모델 생성
    model = UNetKalmanCombined(unet_config, kalman_config, 
                               use_morphology=True, 
                               morphology_kernel_size=5)
    model.eval()  # 평가 모드

    # 테스트 1: 단일 프레임 처리
    print("\n[테스트 1] 단일 프레임 처리")
    print("-" * 60)
    frame = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        result = model(frame, use_kalman=True)
    
    print(f"✓ 입력 프레임: {frame.shape}")
    print(f"✓ 출력 마스크: {result['smoothed_mask'].shape}")
    print(f"✓ 추출된 중심점: {result['center']}")
    print(f"✓ Kalman 예측 중심점: {result['kalman_center']}")
    print(f"✓ 경계박스: {result['bbox']}")

    # 테스트 2: 시퀀셜 프레임 처리
    print("\n[테스트 2] 시퀀셜 프레임 처리 (10 프레임)")
    print("-" * 60)
    model.reset()  # 상태 초기화
    
    frames = [torch.randn(1, 3, 480, 480) for _ in range(10)]
    with torch.no_grad():
        seq_result = model.process_sequence(frames, use_kalman=True)
    
    print(f"✓ 처리한 프레임 수: {model.get_frame_count()}")
    print(f"✓ 생성된 마스크 수: {len(seq_result['smoothed_masks'])}")
    print(f"✓ 추적된 중심점 수: {len(seq_result['centers'])}")
    print(f"✓ 최종 Kalman 상태: {model.get_kalman_state()}")

    # 테스트 3: Kalman Filter 상태 정보
    print("\n[테스트 3] 모델 정보")
    print("-" * 60)
    info = model.get_model_info()
    print(f"✓ 총 프레임 처리: {info['frame_count']}")
    print(f"✓ Kalman 위치: {info['kalman_position']}")
    print(f"✓ Kalman 속도: {info['kalman_velocity']}")

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)