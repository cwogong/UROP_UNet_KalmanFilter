"""
Constant Acceleration (CA) Kalman Filter

Linear KF의 확장 — 등가속도 모델.
기존 Constant Velocity (CV) 모델: [x, y, vx, vy] (4 상태)
등가속도 (CA) 모델: [x, y, vx, vy, ax, ay] (6 상태)

장점:
- 여전히 선형 모델 → 야코비안 불필요, 안정적
- 가속/감속 구간에서 CV보다 정확한 예측
- EKF보다 관측 대비 자유도 적정 (6상태, 2관측)

상태 벡터: [x, y, vx, vy, ax, ay]^T
측정 벡터: [x, y]^T

상태 전이:
    x  = x  + vx*dt + 0.5*ax*dt²
    y  = y  + vy*dt + 0.5*ay*dt²
    vx = vx + ax*dt
    vy = vy + ay*dt
    ax = ax  (일정)
    ay = ay  (일정)
"""

import numpy as np


class ConstantAccelerationFilter:
    """
    2D Constant Acceleration Kalman Filter.

    State: [x, y, vx, vy, ax, ay]^T (6D)
    Measurement: [x, y]^T (2D)
    """

    def __init__(self, dt, x0=None, Q=None, R=None, P=None):
        """
        Args:
            dt (float): Time step
            x0 (np.ndarray): Initial state [x, y, vx, vy, ax, ay] or [x, y, vx, vy]
            Q (np.ndarray): Process noise covariance (6x6) or scalar
            R (np.ndarray): Measurement noise covariance (2x2) or scalar
            P (np.ndarray): Initial state covariance (6x6) or scalar
        """
        self.dt = dt
        self.n_state = 6
        self.n_meas = 2

        # === State Transition Matrix F ===
        # x  = x + vx*dt + 0.5*ax*dt^2
        # y  = y + vy*dt + 0.5*ay*dt^2
        # vx = vx + ax*dt
        # vy = vy + ay*dt
        # ax = ax
        # ay = ay
        dt2 = 0.5 * dt * dt
        self.F = np.array([
            [1, 0, dt, 0,  dt2, 0  ],
            [0, 1, 0,  dt, 0,   dt2],
            [0, 0, 1,  0,  dt,  0  ],
            [0, 0, 0,  1,  0,   dt ],
            [0, 0, 0,  0,  1,   0  ],
            [0, 0, 0,  0,  0,   1  ],
        ], dtype=np.float64)

        # === Observation Matrix H ===
        # 위치만 관측: z = [x, y]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float64)

        # === Process Noise Q ===
        if Q is None:
            self.Q = np.eye(self.n_state, dtype=np.float64) * 0.1
        elif np.isscalar(Q):
            # 물리적으로 의미 있는 Q 구성 (jerk 기반)
            self.Q = self._compute_Q_from_scalar(float(Q), dt)
        elif isinstance(Q, np.ndarray) and Q.shape == (self.n_state, self.n_state):
            self.Q = Q.astype(np.float64)
        else:
            self.Q = np.eye(self.n_state, dtype=np.float64) * float(Q)

        # === Measurement Noise R ===
        if R is None:
            self.R = np.eye(self.n_meas, dtype=np.float64) * 0.5
        elif np.isscalar(R):
            self.R = np.eye(self.n_meas, dtype=np.float64) * float(R)
        elif isinstance(R, np.ndarray) and R.shape == (self.n_meas, self.n_meas):
            self.R = R.astype(np.float64)
        else:
            self.R = np.eye(self.n_meas, dtype=np.float64) * float(R)

        # === State Covariance P ===
        if P is None:
            self.P = np.eye(self.n_state, dtype=np.float64) * 100.0
        elif np.isscalar(P):
            self.P = np.eye(self.n_state, dtype=np.float64) * float(P)
        elif isinstance(P, np.ndarray) and P.shape == (self.n_state, self.n_state):
            self.P = P.astype(np.float64)
        else:
            self.P = np.eye(self.n_state, dtype=np.float64) * float(P)

        # === Initial State ===
        if x0 is None:
            self.x = np.zeros((self.n_state, 1), dtype=np.float64)
        else:
            x0 = np.array(x0, dtype=np.float64).flatten()
            if len(x0) == 4:
                # [x, y, vx, vy] → [x, y, vx, vy, 0, 0]
                x0 = np.array([x0[0], x0[1], x0[2], x0[3], 0.0, 0.0])
            elif len(x0) == 2:
                # [x, y] → [x, y, 0, 0, 0, 0]
                x0 = np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0])
            self.x = x0.reshape(-1, 1)

        # 리셋용 초기값 저장
        self.x_init = self.x.copy()
        self.P_init = self.P.copy()

    def _compute_Q_from_scalar(self, q_scale, dt):
        """
        Jerk-based process noise matrix.
        모델: 가속도가 일정하다 가정, 실제로는 jerk(가가속도)가 노이즈.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt

        # 1D에서의 Q (jerk noise)
        q_1d = np.array([
            [dt5/20, dt4/8, dt3/6],
            [dt4/8,  dt3/3, dt2/2],
            [dt3/6,  dt2/2, dt   ],
        ]) * q_scale

        # 2D로 확장 (x, y 독립)
        Q = np.zeros((6, 6), dtype=np.float64)
        # x, vx, ax
        Q[0, 0] = q_1d[0, 0]
        Q[0, 2] = q_1d[0, 1]
        Q[0, 4] = q_1d[0, 2]
        Q[2, 0] = q_1d[1, 0]
        Q[2, 2] = q_1d[1, 1]
        Q[2, 4] = q_1d[1, 2]
        Q[4, 0] = q_1d[2, 0]
        Q[4, 2] = q_1d[2, 1]
        Q[4, 4] = q_1d[2, 2]
        # y, vy, ay
        Q[1, 1] = q_1d[0, 0]
        Q[1, 3] = q_1d[0, 1]
        Q[1, 5] = q_1d[0, 2]
        Q[3, 1] = q_1d[1, 0]
        Q[3, 3] = q_1d[1, 1]
        Q[3, 5] = q_1d[1, 2]
        Q[5, 1] = q_1d[2, 0]
        Q[5, 3] = q_1d[2, 1]
        Q[5, 5] = q_1d[2, 2]

        return Q

    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        업데이트 단계

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

    # ===== Getter Methods =====

    def get_state(self):
        """Return full state [x, y, vx, vy, ax, ay]"""
        return self.x.flatten()

    def get_position(self):
        """Return position [x, y]"""
        return self.x[:2].flatten()

    def get_velocity(self):
        """Return velocity [vx, vy]"""
        return self.x[2:4].flatten()

    def get_acceleration(self):
        """Return acceleration [ax, ay]"""
        return self.x[4:6].flatten()

    def get_speed(self):
        """Return speed scalar"""
        vel = self.get_velocity()
        return float(np.linalg.norm(vel))

    def get_covariance(self):
        """Return state covariance P"""
        return self.P

    # ===== Setter Methods =====

    def set_state(self, x_new):
        """Update state vector"""
        x_new = np.array(x_new, dtype=np.float64).flatten()
        if len(x_new) == 4:
            x_new = np.array([x_new[0], x_new[1], x_new[2], x_new[3], 0.0, 0.0])
        self.x = x_new.reshape(-1, 1)

    def set_Q(self, Q_new):
        """Update process noise"""
        if np.isscalar(Q_new):
            self.Q = self._compute_Q_from_scalar(float(Q_new), self.dt)
        else:
            self.Q = np.array(Q_new, dtype=np.float64)

    def set_R(self, R_new):
        """Update measurement noise"""
        if np.isscalar(R_new):
            self.R = np.eye(self.n_meas, dtype=np.float64) * float(R_new)
        else:
            self.R = np.array(R_new, dtype=np.float64)

    def reset(self):
        """Reset to initial state"""
        self.x = self.x_init.copy()
        self.P = self.P_init.copy()

    def __str__(self):
        pos = self.get_position()
        vel = self.get_velocity()
        acc = self.get_acceleration()
        return (f"CA Filter State:\n"
                f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}]\n"
                f"  Velocity: [{vel[0]:.4f}, {vel[1]:.4f}]\n"
                f"  Acceleration: [{acc[0]:.4f}, {acc[1]:.4f}]")


# ============================================================
# 테스트
# ============================================================

def main():
    """CA Filter 테스트: 가속/감속 궤적"""
    import matplotlib.pyplot as plt
    from filters.linear_kalman_filter import KalmanFilter as LinearKF

    print("=" * 60)
    print("Phase 4: Constant Acceleration Filter 테스트")
    print("=" * 60)

    dt = 1.0
    num_steps = 100
    noise_std = 5.0

    # === 가속 + 원형 궤적 생성 ===
    ground_truth = np.zeros((num_steps, 2))
    for i in range(num_steps):
        t = i * dt
        if i < 30:
            # 가속 직선
            ground_truth[i, 0] = 0.5 * 0.5 * t**2
            ground_truth[i, 1] = 100
        elif i < 60:
            # 등속 원형
            t_c = (i - 30) * dt
            r = 80
            omega = 0.1
            ground_truth[i, 0] = ground_truth[29, 0] + r * np.sin(omega * t_c)
            ground_truth[i, 1] = 100 + r * (1 - np.cos(omega * t_c))
        else:
            # 감속 직선
            t_d = (i - 60) * dt
            v0 = 8.0
            a = -0.2
            ground_truth[i, 0] = ground_truth[59, 0] + v0 * t_d + 0.5 * a * t_d**2
            ground_truth[i, 1] = ground_truth[59, 1] - 2 * t_d

    measurements = ground_truth + np.random.randn(num_steps, 2) * noise_std

    # === 필터 초기화 ===
    # CV (Linear KF)
    cv_kf = LinearKF(
        dt=dt,
        x0=np.array([measurements[0, 0], measurements[0, 1], 0, 0]),
        Q=np.eye(4) * 0.3,
        R=np.eye(2) * noise_std**2,
    )

    # CA (Constant Acceleration)
    ca_kf = ConstantAccelerationFilter(
        dt=dt,
        x0=np.array([measurements[0, 0], measurements[0, 1], 0, 0, 0, 0]),
        Q=0.5,  # jerk-based Q
        R=np.eye(2) * noise_std**2,
        P=100.0,
    )

    # === 추적 ===
    cv_estimates = []
    ca_estimates = []

    for i in range(num_steps):
        cv_kf.predict()
        cv_kf.update(measurements[i])
        cv_estimates.append(cv_kf.get_position().copy())

        ca_kf.predict()
        ca_kf.update(measurements[i])
        ca_estimates.append(ca_kf.get_position().copy())

    cv_estimates = np.array(cv_estimates)
    ca_estimates = np.array(ca_estimates)

    # === 비교 ===
    cv_errors = np.linalg.norm(cv_estimates - ground_truth, axis=1)
    ca_errors = np.linalg.norm(ca_estimates - ground_truth, axis=1)
    meas_errors = np.linalg.norm(measurements - ground_truth, axis=1)

    print(f"\n📊 성능 비교 (가속+원형+감속 궤적, noise={noise_std}px):")
    print(f"{'Method':<25} {'Mean CLE':<12} {'Max CLE':<12}")
    print("-" * 49)
    print(f"{'Raw Measurement':<25} {np.mean(meas_errors):<12.4f} {np.max(meas_errors):<12.4f}")
    print(f"{'Linear KF (CV)':<25} {np.mean(cv_errors):<12.4f} {np.max(cv_errors):<12.4f}")
    print(f"{'CA Filter':<25} {np.mean(ca_errors):<12.4f} {np.max(ca_errors):<12.4f}")
    print("-" * 49)
    print(f"\nCA vs CV 개선: {np.mean(cv_errors) - np.mean(ca_errors):.4f}px")

    # === 구간별 비교 ===
    print(f"\n📊 구간별 CLE:")
    print(f"{'구간':<20} {'CV':<12} {'CA':<12} {'개선':<12}")
    print("-" * 56)
    for name, s, e in [('가속 (0-30)', 0, 30), ('원형 (30-60)', 30, 60), ('감속 (60-100)', 60, 100)]:
        cv_seg = np.mean(cv_errors[s:e])
        ca_seg = np.mean(ca_errors[s:e])
        print(f"{name:<20} {cv_seg:<12.4f} {ca_seg:<12.4f} {cv_seg - ca_seg:+12.4f}")

    # === 시각화 ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
    axes[0].plot(measurements[:, 0], measurements[:, 1], 'k.', markersize=2, alpha=0.3, label='Measurements')
    axes[0].plot(cv_estimates[:, 0], cv_estimates[:, 1], 'b-', linewidth=1.5, label='CV (Linear KF)')
    axes[0].plot(ca_estimates[:, 0], ca_estimates[:, 1], 'r-', linewidth=1.5, label='CA Filter')
    axes[0].set_title('Trajectory: CV vs CA')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cv_errors, 'b-', label=f'CV (mean={np.mean(cv_errors):.2f})')
    axes[1].plot(ca_errors, 'r-', label=f'CA (mean={np.mean(ca_errors):.2f})')
    axes[1].axvline(30, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(60, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('CLE (pixels)')
    axes[1].set_title('CLE Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/ca_vs_cv_test.png', dpi=150)
    plt.close()
    print(f"\n✓ 시각화 저장: experiments/ca_vs_cv_test.png")

    print(f"\n최종 CA 상태:")
    print(ca_kf)


if __name__ == '__main__':
    main()
