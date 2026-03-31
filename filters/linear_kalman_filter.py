import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    A simple 2D Linear Kalman Filter implementation.
    Assumes a constant velocity model with state [x, y, vx, vy].
    
    Default configuration: 2D constant velocity model
    - State vector: [x, y, vx, vy]^T
    - Measurement vector: [x, y]^T (position only)
    """
    
    def __init__(self, dt, x0=None, Q=None, R=None, P=None):
        """
        Initializes the Kalman Filter with default 2D constant velocity model.

        Args:
            dt (float): Time step (seconds).
            x0 (np.ndarray): Initial state vector [x, y, vx, vy]. 
                           If None, initialized to zeros.
            Q (np.ndarray): Process noise covariance matrix (4x4).
                          If None, uses default value.
            R (np.ndarray): Measurement noise covariance matrix (2x2).
                          If None, uses default value.
            P (np.ndarray): Initial state covariance matrix (4x4).
                          If None, uses default value (100 * I).
        """
        self.dt = dt
        
        # === State Transition Matrix (F) ===
        # Constant velocity model: x_new = x_old + vx*dt
        #                         y_new = y_old + vy*dt
        #                         vx_new = vx_old (constant)
        #                         vy_new = vy_old (constant)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # === Observation Matrix (H) ===
        # We only measure position [x, y], not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # === Process Noise Covariance (Q) ===
        # Default: small value (trust the model)
        if Q is None:
            self.Q = np.eye(4) * 0.01
        else:
            self.Q = Q
        
        # === Measurement Noise Covariance (R) ===
        # Default: moderate uncertainty in measurements
        if R is None:
            self.R = np.eye(2) * 0.5
        else:
            self.R = R
        
        # === Initial State Covariance (P) ===
        # Default: high uncertainty in initial estimate
        if P is None:
            self.P = np.eye(4) * 100.0
        else:
            self.P = P
        
        # === Initial State Vector ===
        if x0 is None:
            self.x = np.zeros((4, 1), dtype=np.float32)
        else:
            self.x = x0.reshape(-1, 1) if x0.ndim == 1 else x0
        
        # Store initial state for reset functionality
        self.x_init = self.x.copy()
        self.P_init = self.P.copy()

    def predict(self):
        """
        Performs the prediction step.
        """
        # State prediction: x_k = F * x_{k-1}
        self.x = np.dot(self.F, self.x)
        # Covariance prediction: P_k = F * P_{k-1} * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        Performs the update step.

        Args:
            z (np.ndarray): The measurement vector.
        """
        # Ensure z is column vector
        z = z.reshape(-1, 1) if z.ndim == 1 else z
        
        # Innovation (measurement residual): y = z - H * x_k
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance: S = H * P_k * H^T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # Kalman Gain: K = P_k * H^T * S^{-1}
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # State update: x_k = x_k + K * y
        self.x = self.x + np.dot(K, y)
        
        # Covariance update: P_k = (I - K * H) * P_k
        I = np.eye(self.F.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
    
    # ===== Getter Methods =====
    
    def get_state(self):
        """Return current state vector [x, y, vx, vy]^T"""
        return self.x.flatten()
    
    def get_position(self):
        """Return current position [x, y]"""
        return self.x[:2].flatten()
    
    def get_velocity(self):
        """Return current velocity [vx, vy]"""
        return self.x[2:4].flatten()
    
    def get_covariance(self):
        """Return current state covariance matrix P"""
        return self.P
    
    # ===== Setter Methods =====
    
    def set_state(self, x_new):
        """Update state vector"""
        self.x = x_new.reshape(-1, 1) if x_new.ndim == 1 else x_new
    
    def set_Q(self, Q_new):
        """Update process noise covariance Q"""
        self.Q = Q_new
    
    def set_R(self, R_new):
        """Update measurement noise covariance R"""
        self.R = R_new
    
    def reset(self):
        """Reset filter to initial state"""
        self.x = self.x_init.copy()
        self.P = self.P_init.copy()
    
    def __str__(self):
        """String representation"""
        pos = self.get_position()
        vel = self.get_velocity()
        return (f"KalmanFilter State:\n"
                f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}]\n"
                f"  Velocity: [{vel[0]:.4f}, {vel[1]:.4f}]")

def main():
    """
    Example: 2D Linear Kalman Filter Test
    """
    dt = 1.0
    num_steps = 100
    
    # === Generate synthetic data ===
    ground_truth = np.zeros((num_steps, 2))
    measurements = np.zeros((num_steps, 2))

    # Straight line trajectory
    for i in range(num_steps):
        ground_truth[i, 0] = 0.5 * i   # x coordinate
        ground_truth[i, 1] = 0.2 * i   # y coordinate

    # Add Gaussian noise to measurements
    noise = np.random.randn(num_steps, 2) * 0.5
    measurements = ground_truth + noise
    
    # === Initialize Kalman Filter ===
    # No need to define F, H, Q, R - they are now built-in!
    x0 = np.array([measurements[0, 0], measurements[0, 1], 0., 0.])
    kf = KalmanFilter(dt=dt, x0=x0)
    
    # === Run Kalman Filter ===
    estimates = []
    for i in range(num_steps):
        kf.predict()
        kf.update(measurements[i, :])
        estimates.append(kf.x[:2].copy())
    
    estimates = np.array(estimates)

    # === Visualization ===
    plt.figure(figsize=(12, 8))
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', label='Ground Truth', linewidth=2)
    plt.plot(measurements[:, 0], measurements[:, 1], 'bo', markersize=3, label='Measurements')
    plt.plot(estimates[:, 0], estimates[:, 1], 'r-', linewidth=2, label='Kalman Filter Estimate')
    
    plt.title('2D Linear Kalman Filter Test')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()
