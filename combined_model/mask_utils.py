"""
마스크 처리 유틸리티

기능:
- 마스크 시각화
- 마스크 메트릭 계산
- 추적 경로 시각화
- 결과 저장/로드
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


class MaskProcessor:
    """마스크 처리 및 시각화 클래스"""
    
    @staticmethod
    def visualize_tracking(frame, mask, center, bbox, 
                          kalman_center=None, title="Tracking Result"):
        """
        추적 결과 시각화
        
        Args:
            frame (np.ndarray): 원본 프레임 (H, W, 3)
            mask (np.ndarray): 세그멘테이션 마스크 (H, W)
            center (tuple): 원본 중심점 (x, y)
            bbox (tuple): 경계박스 (x1, y1, x2, y2)
            kalman_center (tuple): Kalman 필터 예측 중심점 (선택)
            title (str): 그래프 제목
            
        Returns:
            np.ndarray: 시각화된 이미지
        """
        # 프레임 복사
        vis = frame.copy() if frame.dtype == np.uint8 else (frame * 255).astype(np.uint8)
        
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # 마스크 오버레이 (투명도 있음)
        mask_color = mask.astype(np.uint8) * 255
        mask_colored = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # 파란색
        mask_colored[:, :, 1] = 0
        mask_colored[:, :, 2] = 255  # 빨간색
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
        
        # 중심점 표시
        if center is not None:
            cv2.circle(vis, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(center[0]), int(center[1])), 8, (0, 255, 0), 2)
        
        # Kalman 필터 예측 중심점 표시
        if kalman_center is not None:
            cv2.circle(vis, (int(kalman_center[0]), int(kalman_center[1])), 
                      5, (255, 0, 0), -1)
            cv2.circle(vis, (int(kalman_center[0]), int(kalman_center[1])), 
                      8, (255, 0, 0), 2)
        
        # 경계박스 표시
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # 제목 추가
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        
        # 범례
        cv2.putText(vis, "Green: Detected Center", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if kalman_center is not None:
            cv2.putText(vis, "Blue: Kalman Predicted", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return vis

    @staticmethod
    def plot_tracking_trajectory(centers, kalman_centers, 
                                save_path=None, show=True):
        """
        추적 궤적 시각화
        
        Args:
            centers (list): 추출된 중심점 리스트
            kalman_centers (list): Kalman 필터 예측 중심점 리스트
            save_path (str): 저장 경로 (선택)
            show (bool): 표시 여부
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 데이터 정제
        centers_valid = [c for c in centers if c is not None]
        kalman_centers_valid = [k for k in kalman_centers if k is not None]
        
        if len(centers_valid) > 0:
            centers_array = np.array(centers_valid)
            ax.plot(centers_array[:, 0], centers_array[:, 1], 'go-', 
                   label='Detected', linewidth=2, markersize=6)
        
        if len(kalman_centers_valid) > 0:
            kalman_array = np.array(kalman_centers_valid)
            ax.plot(kalman_array[:, 0], kalman_array[:, 1], 'b^-', 
                   label='Kalman Filtered', linewidth=2, markersize=6)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Object Tracking Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 궤적 이미지 저장: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()

    @staticmethod
    def calculate_metrics(detected_centers, kalman_centers):
        """
        추적 메트릭 계산
        
        Args:
            detected_centers (list): 감지된 중심점 리스트
            kalman_centers (list): Kalman 필터 중심점 리스트
            
        Returns:
            dict: 메트릭 (거리, 평활화 효과 등)
        """
        # 데이터 정제
        detected_valid = np.array([c for c in detected_centers if c is not None])
        kalman_valid = np.array([k for k in kalman_centers if k is not None])
        
        if len(detected_valid) == 0 or len(kalman_valid) == 0:
            return {'error': 'Insufficient data'}
        
        # Kalman 필터와 감지의 차이
        min_len = min(len(detected_valid), len(kalman_valid))
        detected_valid = detected_valid[:min_len]
        kalman_valid = kalman_valid[:min_len]
        
        # 거리 계산
        distances = np.linalg.norm(detected_valid - kalman_valid, axis=1)
        
        # 속도 계산
        if len(detected_valid) > 1:
            detected_velocity = np.diff(detected_valid, axis=0)
            kalman_velocity = np.diff(kalman_valid, axis=0)
            
            detected_speed = np.linalg.norm(detected_velocity, axis=1)
            kalman_speed = np.linalg.norm(kalman_velocity, axis=1)
        else:
            detected_speed = []
            kalman_speed = []
        
        return {
            'num_frames': len(detected_valid),
            'avg_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'detected_avg_speed': np.mean(detected_speed) if len(detected_speed) > 0 else 0,
            'kalman_avg_speed': np.mean(kalman_speed) if len(kalman_speed) > 0 else 0,
            'smoothness_improvement': (np.std(detected_speed) - np.std(kalman_speed)) / 
                                      (np.std(detected_speed) + 1e-6) * 100
        }

    @staticmethod
    def save_video(frames, masks, centers, bboxes, output_path, fps=30):
        """
        추적 결과를 비디오로 저장
        
        Args:
            frames (list): 원본 프레임 리스트
            masks (list): 마스크 리스트
            centers (list): 중심점 리스트
            bboxes (list): 경계박스 리스트
            output_path (str): 출력 경로
            fps (int): 프레임 레이트
        """
        if len(frames) == 0:
            print("Error: 빈 프레임 리스트")
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for i, frame in enumerate(frames):
            mask = masks[i] if i < len(masks) else None
            center = centers[i] if i < len(centers) else None
            bbox = bboxes[i] if i < len(bboxes) else None
            
            vis = MaskProcessor.visualize_tracking(
                frame, mask, center, bbox, 
                title=f"Frame {i+1}"
            )
            
            writer.write(vis)
        
        writer.release()
        print(f"✓ 비디오 저장: {output_path}")


class TrackingVisualizer:
    """추적 결과 시각화 클래스"""
    
    def __init__(self, save_dir='./tracking_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_frame_result(self, frame, result, frame_idx):
        """
        개별 프레임 결과 저장
        
        Args:
            frame (torch.Tensor 또는 np.ndarray): 입력 프레임
            result (dict): forward() 반환 결과
            frame_idx (int): 프레임 인덱스
        """
        # 프레임을 numpy로 변환
        if hasattr(frame, 'cpu'):
            frame_np = frame.squeeze().cpu().numpy()
            if frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
        else:
            frame_np = frame
        
        # 마스크를 numpy로 변환
        if hasattr(result['smoothed_mask'], 'cpu'):
            mask_np = result['smoothed_mask'].squeeze().cpu().numpy()
        else:
            mask_np = result['smoothed_mask'].squeeze()
        
        # 시각화
        vis = MaskProcessor.visualize_tracking(
            frame_np, mask_np, result['center'], 
            result['bbox'], result['kalman_center']
        )
        
        # 저장
        save_path = self.save_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR) if len(vis.shape) == 3 else vis)
        
        return save_path

    def generate_report(self, results_list, output_path=None):
        """
        추적 결과 리포트 생성
        
        Args:
            results_list (list): 프레임별 결과 리스트
            output_path (str): 저장 경로 (선택)
        """
        centers = [r['center'] for r in results_list]
        kalman_centers = [r['kalman_center'] for r in results_list]
        
        metrics = MaskProcessor.calculate_metrics(centers, kalman_centers)
        
        report = f"""
╔═════════════════════════════════════════════════════════════════╗
║          Phase 2: UNet + Kalman Filter 추적 결과 리포트           ║
╚═════════════════════════════════════════════════════════════════╝

📊 처리된 프레임 수: {metrics.get('num_frames', 'N/A')}

📏 중심점 거리 분석:
   - 평균 거리: {metrics.get('avg_distance', 0):.2f} pixels
   - 최대 거리: {metrics.get('max_distance', 0):.2f} pixels
   - 표준편차: {metrics.get('std_distance', 0):.2f} pixels

🚀 속도 분석:
   - 감지된 평균 속도: {metrics.get('detected_avg_speed', 0):.2f} pixels/frame
   - Kalman 필터 평균 속도: {metrics.get('kalman_avg_speed', 0):.2f} pixels/frame
   - 평활도 개선: {metrics.get('smoothness_improvement', 0):.2f}%

✅ 결과:
   - 마스크 이미지: {self.save_dir / 'frame_*.png'}
   - 궤적 그래프: {self.save_dir / 'trajectory.png'}
"""
        
        print(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)


if __name__ == '__main__':
    print("마스크 처리 유틸리티 모듈입니다.")
    print("사용: from combined_model.mask_utils import MaskProcessor, TrackingVisualizer")
