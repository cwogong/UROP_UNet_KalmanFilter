"""
Extended Kalman Filter (EKF) — CTRV 모션 모델

모션 모델: Constant Turn Rate and Velocity (CTRV)
- UAV의 속도(speed)와 회전율(yaw_rate)이 일정하다고 가정
- 등속도 모델보다 곡선 궤적, 선회 운동을 잘 모델링

상태 벡터: [x, y, v, θ, ω]
- x, y: 위치
- v: 속도 크기 (speed)
- θ (theta): 진행 방향 (heading, radian)
- ω (omega): 회전 속도 (yaw rate, rad/s)

측정 벡터: [x, y] (위치만 관측)

비선형 상태 전이:
    x_new = x + v/ω * (sin(θ + ω*dt) - sin(θ))     [ω ≠ 0]
    y_new = y + v/ω * (-cos(θ + ω*dt) + cos(θ))     [ω ≠ 0]
    v_new = v
    θ_new = θ + ω*dt
    ω_new = ω

    (ω ≈ 0일 때)
    x_new = x + v*cos(θ)*dt
    y_new = y + v*sin(θ)*dt
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with CTRV motion model.

    State: [x, y, v, theta, omega]^T (5D)
    Measurement: [x, y]^T (2D)
    """

    def __init__(self, dt, x0=None, Q=None, R=None, P=None):
        """
        Args:
            dt (float): Time step (seconds)
            x0 (np.ndarray): Initial state [x, y, v, theta, omega]
                            If None, defaults to [0, 0, 0, 0, 0]
            Q (np.ndarray): Process noise covariance (5x5)
            R (np.ndarray): Measurement noise covariance (2x2)
            P (np.ndarray): Initial state covariance (5x5)
        """
        self.dt = dt
        self.n_state = 5
        self.n_meas = 2

        # 상태 초기화
        if x0 is None:
            self.x = np.zeros((self.n_state, 1), dtype=np.float64)
        else:
            x0 = np.array(x0, dtype=np.float64).flatten()
            if len(x0) == 4:
                # [x, y, vx, vy] → [x, y, speed, heading, 0]
                vx, vy = x0[2], x0[3]
                speed = np.sqrt(vx**2 + vy**2)
                theta = np.arctan2(vy, vx)
                x0 = np.array([x0[0], x0[1], speed, theta, 0.0])
            self.x = x0.reshape(-1, 1)

        # Process noise
        if Q is None:
            self.Q = np.diag([1.0, 1.0, 0.5, 0.1, 0.1]).astype(np.float64)
        else:
            self.Q = np.array(Q, dtype=np.float64)
            if self.Q.shape == ():
                self.Q = np.eye(self.n_state) * float(self.Q)
            elif self.Q.shape == (self.n_state,):
                self.Q = np.diag(self.Q)

        # Measurement noise
        if R is None:
            self.R = np.eye(self.n_meas, dtype=np.float64) * 0.5
        else:
            self.R = np.array(R, dtype=np.float64)
            if self.R.shape == ():
                self.R = np.eye(self.n_meas) * float(self.R)

        # State covariance
        if P is None:
            self.P = np.eye(self.n_state, dtype=np.float64) * 100.0
        else:
            self.P = np.array(P, dtype=np.float64)
            if self.P.shape == ():
                self.P = np.eye(self.n_state) * float(self.P)

        # Observation matrix (linear: observe x, y only)
        self.H = np.zeros((self.n_meas, self.n_state), dtype=np.float64)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y

        # 초기값 저장 (리셋용)
        self.x_init = self.x.copy()
        self.P_init = self.P.copy()

    def _f(self, x, dt):
        """
        비선형 상태 전이 함수 f(x)

        CTRV 모델:
        - ω ≠ 0: 곡선 운동
        - ω ≈ 0: 직선 운동 (특이점 회피)
        """
        px, py, v, theta, omega = x.flatten()

        if abs(omega) > 1e-5:
            # 곡선 운동
            px_new = px + v / omega * (np.sin(theta + omega * dt) - np.sin(theta))
            py_new = py + v / omega * (-np.cos(theta + omega * dt) + np.cos(theta))
        else:
            # 직선 운동 (ω ≈ 0)
            px_new = px + v * np.cos(theta) * dt
            py_new = py + v * np.sin(theta) * dt

        v_new = v
        theta_new = theta + omega * dt
        omega_new = omega

        # 각도 정규화 [-π, π]
        theta_new = self._normalize_angle(theta_new)

        return np.array([[px_new], [py_new], [v_new], [theta_new], [omega_new]],
                       dtype=np.float64)

    def _jacobian_F(self, x, dt):
        """
        상태 전이 야코비안 ∂f/∂x

        Returns:
            np.ndarray: (5x5) Jacobian matrix
        """
        px, py, v, theta, omega = x.flatten()

        F = np.eye(self.n_state, dtype=np.float64)

        if abs(omega) > 1e-5:
            # ∂f/∂x에 대한 편미분
            s_t = np.sin(theta)
            c_t = np.cos(theta)
            s_tw = np.sin(theta + omega * dt)
            c_tw = np.cos(theta + omega * dt)

            # ∂px/∂v
            F[0, 2] = (s_tw - s_t) / omega
            # ∂px/∂theta
            F[0, 3] = v / omega * (c_tw - c_t)
            # ∂px/∂omega
            F[0, 4] = v / omega * (c_tw * dt) - v / (omega**2) * (s_tw - s_t)

            # ∂py/∂v
            F[1, 2] = (-c_tw + c_t) / omega
            # ∂py/∂theta
            F[1, 3] = v / omega * (s_tw - s_t)
            # ∂py/∂omega
            F[1, 4] = v / omega * (s_tw * dt) - v / (omega**2) * (-c_tw + c_t)
        else:
            # 직선 운동
            c_t = np.cos(theta)
            s_t = np.sin(theta)

            # ∂px/∂v
            F[0, 2] = c_t * dt
            # ∂px/∂theta
            F[0, 3] = -v * s_t * dt

            # ∂py/∂v
            F[1, 2] = s_t * dt
            # ∂py/∂theta
            F[1, 3] = v * c_t * dt

        # ∂theta/∂omega
        F[3, 4] = dt

        return F

    def predict(self):
        """EKF 예측 단계"""
        # 비선형 상태 전이
        self.x = self._f(self.x, self.dt)

        # 야코비안 계산
        F = self._jacobian_F(self.x, self.dt)

        # 공분산 예측
        self.P = F @ self.P @ F.T + self.Q

        return self.x

    def update(self, z):
        """
        EKF 업데이트 단계

        Args:
            z (np.ndarray): 측정값 [x, y]
        """
        z = np.array(z, dtype=np.float64).reshape(-1, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.n_state)
        self.P = (I - K @ self.H) @ self.P

        # 각도 정규화
        self.x[3, 0] = self._normalize_angle(self.x[3, 0])

    # ===== Getter Methods =====

    def get_state(self):
        """Return full state [x, y, v, theta, omega]"""
        return self.x.flatten()

    def get_position(self):
        """Return position [x, y]"""
        return self.x[:2].flatten()

    def get_velocity(self):
        """Return velocity as [vx, vy] (for compatibility with linear KF)"""
        v = self.x[2, 0]
        theta = self.x[3, 0]
        return np.array([v * np.cos(theta), v * np.sin(theta)])

    def get_speed(self):
        """Return speed scalar"""
        return float(self.x[2, 0])

    def get_heading(self):
        """Return heading angle (radians)"""
        return float(self.x[3, 0])

    def get_yaw_rate(self):
        """Return yaw rate (rad/s)"""
        return float(self.x[4, 0])

    def get_covariance(self):
        """Return state covariance P"""
        return self.P

    # ===== Setter Methods =====

    def set_state(self, x_new):
        """Update state vector"""
        x_new = np.array(x_new, dtype=np.float64).flatten()
        if len(x_new) == 4:
            vx, vy = x_new[2], x_new[3]
            speed = np.sqrt(vx**2 + vy**2)
            theta = np.arctan2(vy, vx)
            x_new = np.array([x_new[0], x_new[1], speed, theta, 0.0])
        self.x = x_new.reshape(-1, 1)

    def set_Q(self, Q_new):
        """Update process noise"""
        self.Q = np.array(Q_new, dtype=np.float64)
        if self.Q.shape == ():
            self.Q = np.eye(self.n_state) * float(self.Q)

    def set_R(self, R_new):
        """Update measurement noise"""
        self.R = np.array(R_new, dtype=np.float64)
        if self.R.shape == ():
            self.R = np.eye(self.n_meas) * float(self.R)

    def reset(self):
        """Reset to initial state"""
        self.x = self.x_init.copy()
        self.P = self.P_init.copy()

    # ===== Utility =====

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def __str__(self):
        pos = self.get_position()
        vel = self.get_velocity()
        return (f"EKF (CTRV) State:\n"
                f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}]\n"
                f"  Speed: {self.get_speed():.4f}\n"
                f"  Heading: {np.degrees(self.get_heading()):.1f}°\n"
                f"  Yaw Rate: {np.degrees(self.get_yaw_rate()):.1f}°/s")


# ============================================================
# 테스트
# ============================================================

def main():
    """EKF CTRV 모델 테스트: 원형 궤적 추적"""
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Phase 4: Extended Kalman Filter (CTRV) 테스트")
    print("=" * 60)

    dt = 1.0
    num_steps = 100

    # === 원형 궤적 생성 ===
    radius = 100.0
    omega_true = 2 * np.pi / num_steps  # 1 revolution
    speed_true = radius * omega_true

    ground_truth = np.zeros((num_steps, 2))
    measurements = np.zeros((num_steps, 2))

    cx, cy = 240.0, 240.0
    for i in range(num_steps):
        t = omega_true * i * dt
        ground_truth[i, 0] = cx + radius * np.cos(t)
        ground_truth[i, 1] = cy + radius * np.sin(t)

    # 측정 노이즈 추가
    noise_std = 5.0
    measurements = ground_truth + np.random.randn(num_steps, 2) * noise_std

    # === EKF 초기화 ===
    # 초기 위치에서 방향 추정
    init_theta = np.arctan2(
        ground_truth[1, 1] - ground_truth[0, 1],
        ground_truth[1, 0] - ground_truth[0, 0]
    )

    x0 = np.array([measurements[0, 0], measurements[0, 1],
                    speed_true, init_theta, omega_true])

    ekf = ExtendedKalmanFilter(
        dt=dt,
        x0=x0,
        Q=np.diag([0.5, 0.5, 0.1, 0.01, 0.01]),
        R=np.eye(2) * noise_std**2,
        P=np.eye(5) * 10.0,
    )

    # === 선형 Kalman Filter (비교용) ===
    from filters.linear_kalman_filter import KalmanFilter as LinearKF

    lkf = LinearKF(
        dt=dt,
        x0=np.array([measurements[0, 0], measurements[0, 1], 0, 0]),
        Q=np.eye(4) * 0.1,
        R=np.eye(2) * noise_std**2,
    )

    # === 추적 실행 ===
    ekf_estimates = []
    lkf_estimates = []

    for i in range(num_steps):
        # EKF
        ekf.predict()
        ekf.update(measurements[i])
        ekf_estimates.append(ekf.get_position().copy())

        # Linear KF
        lkf.predict()
        lkf.update(measurements[i])
        lkf_estimates.append(lkf.get_position().copy())

    ekf_estimates = np.array(ekf_estimates)
    lkf_estimates = np.array(lkf_estimates)

    # === 성능 비교 ===
    ekf_errors = np.linalg.norm(ekf_estimates - ground_truth, axis=1)
    lkf_errors = np.linalg.norm(lkf_estimates - ground_truth, axis=1)
    meas_errors = np.linalg.norm(measurements - ground_truth, axis=1)

    print(f"\n📊 성능 비교 (원형 궤적, noise_std={noise_std}px):")
    print(f"{'Method':<20} {'Mean CLE':<12} {'Max CLE':<12} {'Std CLE':<12}")
    print("-" * 56)
    print(f"{'Raw Measurement':<20} {np.mean(meas_errors):<12.4f} {np.max(meas_errors):<12.4f} {np.std(meas_errors):<12.4f}")
    print(f"{'Linear KF':<20} {np.mean(lkf_errors):<12.4f} {np.max(lkf_errors):<12.4f} {np.std(lkf_errors):<12.4f}")
    print(f"{'EKF (CTRV)':<20} {np.mean(ekf_errors):<12.4f} {np.max(ekf_errors):<12.4f} {np.std(ekf_errors):<12.4f}")
    print("-" * 56)
    print(f"\nEKF vs Linear KF 개선: {(np.mean(lkf_errors) - np.mean(ekf_errors)):.4f}px")

    # === 시각화 ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 궤적 비교
    axes[0].plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
    axes[0].plot(measurements[:, 0], measurements[:, 1], 'k.', markersize=2, alpha=0.3, label='Measurements')
    axes[0].plot(lkf_estimates[:, 0], lkf_estimates[:, 1], 'b-', linewidth=1.5, label='Linear KF')
    axes[0].plot(ekf_estimates[:, 0], ekf_estimates[:, 1], 'r-', linewidth=1.5, label='EKF (CTRV)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Trajectory Comparison')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # CLE 시계열
    axes[1].plot(meas_errors, 'k-', alpha=0.3, label='Raw Measurement')
    axes[1].plot(lkf_errors, 'b-', label=f'Linear KF (mean={np.mean(lkf_errors):.2f})')
    axes[1].plot(ekf_errors, 'r-', label=f'EKF CTRV (mean={np.mean(ekf_errors):.2f})')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('CLE (pixels)')
    axes[1].set_title('Center Location Error Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ekf_vs_lkf_test.png', dpi=150)
    plt.close()
    print(f"\n✓ 시각화 저장: experiments/ekf_vs_lkf_test.png")

    # EKF 상태 출력
    print(f"\n최종 EKF 상태:")
    print(ekf)


if __name__ == '__main__':
    main()
