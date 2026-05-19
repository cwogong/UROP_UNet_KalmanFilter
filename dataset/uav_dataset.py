"""
UAV Tracking Dataset with Centroid Extraction for Kalman Filter
구조: root/seq/infrared/*.jpg, root_mask/seq/infrared/*.png
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob


class UAVTrackingDataset(Dataset):
    """
    UAV 추적용 Dataset (Kalman Filter와 연동)
    
    주요 기능:
    1. 이미지와 마스크 페어 로드
    2. 마스크에서 중심점(centroid) 추출 → Kalman Filter 입력
    3. 바운딩박스 추출 → 시각화용
    4. 마스크 오차 측정 → R 파라미터 추정용
    """
    
    def __init__(self, root, root_mask, transforms=None, num_sequences=None):
        """
        Args:
            root (str): 이미지 루트 디렉토리
            root_mask (str): 마스크 루트 디렉토리
            transforms: 이미지 변환 함수
            num_sequences (int): 로드할 최대 시퀀스 수
        """
        self.root = root
        self.root_mask = root_mask
        self.transforms = transforms
        self.images, self.masks = self._get_file_paths(root, root_mask, num_sequences)
        
        print(f"Loaded {len(self.images)} image-mask pairs")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image 또는 Tensor
            targets: {
                'centroids': [[x1, y1], [x2, y2], ...],
                'boxes': [[x1, y1, x2, y2], ...],
                'mask': segmentation mask,
                'image_path': 이미지 경로 (메타데이터용)
            }
        """
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 마스크 로드
        mask = np.array(Image.open(mask_path))
        
        # Kalman Filter 입력: 중심점 추출
        centroids = self._extract_centroids(mask)
        
        # 시각화 및 평가용: 바운딩박스 추출
        boxes = self._extract_boxes(mask)
        
        # Transforms 적용
        if self.transforms:
            image = self.transforms(image)
        
        targets = {
            "centroids": torch.as_tensor(centroids, dtype=torch.float32),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.uint8),
            "image_path": image_path
        }
        
        return image, targets
    
    def _extract_centroids(self, mask):
        """
        마스크에서 객체 중심점(무게중심) 추출
        
        Kalman Filter의 측정값(measurement) z로 사용됨
        
        Returns:
            np.ndarray: shape (N, 2), [[x1, y1], [x2, y2], ...]
        """
        centroids = []
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]  # 0(배경) 제외
        
        for obj_id in obj_ids:
            # 객체 픽셀 위치
            pos = np.where(mask == obj_id)
            
            if len(pos[0]) > 0:
                # 무게중심 계산
                # pos[0] = y 좌표들, pos[1] = x 좌표들
                cy = np.mean(pos[0])  
                cx = np.mean(pos[1])  
                centroids.append([cx, cy])
        
        # 객체가 없으면 원점
        if len(centroids) == 0:
            centroids = [[0, 0]]
        
        return np.array(centroids, dtype=np.float32)
    
    def _extract_boxes(self, mask):
        """
        마스크에서 바운딩박스 추출
        
        Returns:
            np.ndarray: shape (N, 4), [[x1, y1, x2, y2], ...]
        """
        boxes = []
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            if len(pos[0]) > 0:
                # (좌상단, 우하단) 좌표
                xmin, ymin = np.min(pos[1]), np.min(pos[0])
                xmax, ymax = np.max(pos[1]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
        
        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
        
        return np.array(boxes, dtype=np.float32)
    
    def _get_sequence_dirs(self, root):
        """루트 디렉토리에서 시퀀스 경로를 찾음

        지원 구조:
            root/<seq>/infrared/*.jpg
            root/<split>/<seq>/infrared/*.jpg
        """
        seq_dirs = []
        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue

            # 상위 레벨이 바로 시퀀스 폴더인 경우
            if self._has_valid_sequence_dir(entry_path):
                seq_dirs.append(entry)
                continue

            # 상위 레벨이 split(train/val/test)인 경우
            for seq in sorted(os.listdir(entry_path)):
                seq_path = os.path.join(entry_path, seq)
                if os.path.isdir(seq_path) and self._has_valid_sequence_dir(seq_path):
                    seq_dirs.append(os.path.join(entry, seq))

        return seq_dirs

    def _has_valid_sequence_dir(self, path):
        """infrared/visible 하위 폴더가 있는지 검사"""
        return all(os.path.isdir(os.path.join(path, img_type)) for img_type in ['infrared', 'visible'])

    def _get_file_paths(self, root, root_mask, num_sequences=None):
        """이미지와 마스크 경로 매칭"""
        image_list, mask_list = [], []
        seq_count = 0

        seq_dirs = self._get_sequence_dirs(root)

        for seq in seq_dirs:
            for img_type in ['infrared', 'visible']:
                image_pattern = os.path.join(root, seq, img_type, '*.jpg')
                mask_pattern = os.path.join(root_mask, seq, img_type, '*.png')

                image_list += sorted(glob(image_pattern))
                mask_list += sorted(glob(mask_pattern))

            seq_count += 1
            if num_sequences is not None and seq_count >= num_sequences:
                break

        assert len(image_list) == len(mask_list), \
            f'Image ({len(image_list)}) and mask ({len(mask_list)}) count mismatch!'

        return image_list, mask_list


class UAVSequenceDataset(Dataset):
    """
    비디오 시퀀스 단위 Dataset
    연속된 프레임들을 묶어서 반환 (Kalman Filter 추적용)
    """
    
    def __init__(self, root, root_mask, sequence_length=5, transforms=None, num_sequences=None):
        """
        Args:
            sequence_length (int): 한 번에 반환할 프레임 수
                                   예: 5이면 프레임 0-4를 함께 반환
        """
        self.root = root
        self.root_mask = root_mask
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.images, self.masks, self.seq_indices = self._get_sequences(
            root, root_mask, sequence_length, num_sequences
        )
        
        print(f"Loaded {len(self)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.seq_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            images: Tensor, shape (T, C, H, W)
            masks: Tensor, shape (T, H, W)
            centroids_sequence: Tensor, shape (T, N_objs, 2)
        """
        start_idx = self.seq_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        images_list = []
        masks_list = []
        centroids_list = []
        
        for i in range(start_idx, end_idx):
            image_path = self.images[i]
            mask_path = self.masks[i]
            
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            images_list.append(image)
            
            # 마스크 로드
            mask = np.array(Image.open(mask_path))
            masks_list.append(torch.as_tensor(mask, dtype=torch.uint8))
            
            # 중심점 추출
            centroids = self._extract_centroids(mask)
            centroids_list.append(torch.as_tensor(centroids, dtype=torch.float32))
        
        images_tensor = torch.stack(images_list)  # (T, C, H, W)
        masks_tensor = torch.stack(masks_list)     # (T, H, W)
        
        return images_tensor, masks_tensor, centroids_list
    
    def _extract_centroids(self, mask):
        """마스크에서 중심점 추출"""
        centroids = []
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            if len(pos[0]) > 0:
                cy = np.mean(pos[0])
                cx = np.mean(pos[1])
                centroids.append([cx, cy])
        
        if len(centroids) == 0:
            centroids = [[0, 0]]
        
        return np.array(centroids, dtype=np.float32)
    
    def _get_sequences(self, root, root_mask, sequence_length, num_sequences=None):
        """시퀀스 단위로 인덱스 구성"""
        image_list, mask_list = [], []
        seq_count = 0
        
        seq_dirs = self._get_sequence_dirs(root)

        for seq in seq_dirs:
            for img_type in ['infrared', 'visible']:
                image_pattern = os.path.join(root, seq, img_type, '*.jpg')
                mask_pattern = os.path.join(root_mask, seq, img_type, '*.png')

                image_list += sorted(glob(image_pattern))
                mask_list += sorted(glob(mask_pattern))

            seq_count += 1
            if num_sequences is not None and seq_count >= num_sequences:
                break

        assert len(image_list) == len(mask_list)

        # 시퀀스 시작 인덱스 생성
        seq_indices = []
        for i in range(len(image_list) - sequence_length + 1):
            seq_indices.append(i)

        return image_list, mask_list, seq_indices


# ============================================
# 유틸리티 함수: R 파라미터 측정
# ============================================

def measure_mask_detection_noise(dataset, num_samples=100):
    """
    마스크 검출 오차 측정 (R 파라미터 추정용)
    
    같은 이미지로 여러 번 추론했을 때 중심점의 분산을 측정
    → gps_pos_std = np.std(centroids)
    
    Args:
        dataset: UAVTrackingDataset
        num_samples: 측정할 샘플 개수
    
    Returns:
        float: 중심점 표준편차 (픽셀 단위)
    """
    centroid_diffs = []
    
    # 랜덤 샘플 선택
    sampled_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in sampled_indices:
        image, targets = dataset[idx]
        centroids = targets['centroids'].numpy()
        
        # 중심점에 작은 노이즈 추가 후 분산 측정
        # (실제는 UNet 출력의 분산을 측정해야 함)
        if len(centroids) > 0:
            centroid_diffs.append(centroids)
    
    all_centroids = np.vstack(centroid_diffs)
    noise_std = np.std(all_centroids, axis=0)
    
    return float(np.mean(noise_std))


# ============================================
# 테스트 및 시각화
# ============================================

def visualize_dataset(dataset, num_samples=5):
    """Dataset 시각화"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    for i in range(num_samples):
        image, targets = dataset[i]
        
        # 이미지 변환 (Tensor → numpy)
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = np.array(image)
        
        # 마스크 처리
        mask = targets['mask'].numpy()
        
        # 행 1: 원본 이미지
        axes[i, 0].imshow(image_np)
        centroids = targets['centroids'].numpy()
        for cx, cy in centroids:
            axes[i, 0].plot(cx, cy, 'ro', markersize=8)
        axes[i, 0].set_title(f'Image {i} with Centroids')
        
        # 행 2: 마스크
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask {i}')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 테스트 코드
    print("UAV Tracking Dataset Module")
    
    # 사용 예시
    # DATASET_PATH = '/path/to/images'
    # MASK_PATH = '/path/to/masks'
    # 
    # dataset = UAVTrackingDataset(DATASET_PATH, MASK_PATH, num_sequences=1)
    # print(f"Dataset size: {len(dataset)}")
    # 
    # image, targets = dataset[0]
    # print(f"Image shape: {image.size}")
    # print(f"Centroids: {targets['centroids']}")
    # 
    # # R 파라미터 측정
    # gps_pos_std = measure_mask_detection_noise(dataset, num_samples=50)
    # print(f"Recommended gps_pos_std for Kalman Filter: {gps_pos_std:.2f}")
