"""
Phase 1: Linear Kalman Filter Independent Implementation and Testing
Goal: Basic functionality verification and performance evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from filters.linear_kalman_filter import KalmanFilter


def ensure_output_dir():
    """Create experiments directory if it doesn't exist"""
    os.makedirs('experiments', exist_ok=True)


def test_1d_constant_velocity():
    """
    Test 1: 1D Constant Velocity Motion Tracking
    Using 2D filter with y=0, velocity=[vx, 0]
    """
    print("\n" + "="*60)
    print("Test 1: 1D Constant Velocity Motion Tracking")
    print("="*60)

    dt = 1.0
    num_steps = 50

    # Real trajectory (constant velocity)
    true_positions = np.arange(num_steps, dtype=np.float32)
    
    # Measurements with noise
    np.random.seed(42)
    noise = np.random.randn(num_steps) * 0.2
    measurements = true_positions + noise

    # Initialize Kalman Filter
    x0 = np.array([measurements[0], 0.0, 1.0, 0.0])  # [x, y, vx, vy]
    kf = KalmanFilter(dt=dt, x0=x0)

    # Tracking loop
    estimates = []
    for i in range(num_steps):
        kf.predict()
        # Update with 2D measurement [x, y] but only x changes
        kf.update(np.array([measurements[i], 0.0]))
        est_pos = kf.get_position()
        estimates.append(est_pos[0])  # Extract x position

    estimates = np.array(estimates)

    # Performance evaluation
    measurement_error = measurements - true_positions
    filter_error = estimates - true_positions

    measurement_rmse = np.sqrt(np.mean(measurement_error**2))
    filter_rmse = np.sqrt(np.mean(filter_error**2))
    improvement = ((measurement_rmse - filter_rmse) / measurement_rmse * 100) if measurement_rmse > 0 else 0

    print(f"Measurement RMSE: {measurement_rmse:.4f}")
    print(f"Filter RMSE:      {filter_rmse:.4f}")
    print(f"Improvement:      {improvement:.2f}%")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Position tracking
    axes[0].plot(true_positions, 'g-', label='True Position', linewidth=2.5, marker='o', markersize=3)
    axes[0].plot(measurements, 'r.', label='Measurements', alpha=0.6, markersize=4)
    axes[0].plot(estimates, 'b-', label='Kalman Estimate', linewidth=2.5, marker='s', markersize=3)
    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Position (x)', fontsize=11)
    axes[0].set_title('Test 1: 1D Constant Velocity Tracking', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Error comparison
    axes[1].plot(measurement_error, 'r-', label='Measurement Error', alpha=0.7, linewidth=2)
    axes[1].plot(filter_error, 'b-', label='Filter Error', linewidth=2.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].fill_between(range(num_steps), measurement_error, alpha=0.2, color='red')
    axes[1].fill_between(range(num_steps), filter_error, alpha=0.2, color='blue')
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Error Comparison', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/test_1_1d_tracking.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: experiments/test_1_1d_tracking.png")

    return {
        'name': 'Test 1: 1D Constant Velocity',
        'measurement_rmse': measurement_rmse,
        'filter_rmse': filter_rmse,
        'improvement': improvement
    }


def test_2d_straight_line():
    """
    Test 2: 2D Straight Line Motion Tracking
    """
    print("\n" + "="*60)
    print("Test 2: 2D Straight Line Motion Tracking")
    print("="*60)

    dt = 1.0
    num_steps = 100

    # Real trajectory (diagonal motion)
    true_positions = np.array([[0.5*i, 0.3*i] for i in range(num_steps)])

    # Measurements with noise
    np.random.seed(43)
    noise = np.random.randn(num_steps, 2) * 0.2
    measurements = true_positions + noise

    # Initialize Kalman Filter
    x0 = np.array([measurements[0, 0], measurements[0, 1], 0.5, 0.3])
    kf = KalmanFilter(dt=dt, x0=x0)

    # Tracking loop
    estimates = []
    for i in range(num_steps):
        kf.predict()
        kf.update(measurements[i, :])
        estimates.append(kf.get_position())

    estimates = np.array(estimates)

    # Performance evaluation
    measurement_error = np.linalg.norm(measurements - true_positions, axis=1)
    filter_error = np.linalg.norm(estimates - true_positions, axis=1)

    measurement_rmse = np.sqrt(np.mean(measurement_error**2))
    filter_rmse = np.sqrt(np.mean(filter_error**2))
    improvement = ((measurement_rmse - filter_rmse) / measurement_rmse * 100) if measurement_rmse > 0 else 0

    print(f"Measurement RMSE: {measurement_rmse:.4f}")
    print(f"Filter RMSE:      {filter_rmse:.4f}")
    print(f"Improvement:      {improvement:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D Trajectory
    axes[0].plot(true_positions[:, 0], true_positions[:, 1], 'g-', 
                 label='True Trajectory', linewidth=2.5, marker='o', markersize=2)
    axes[0].plot(measurements[:, 0], measurements[:, 1], 'r.', 
                 label='Measurements', alpha=0.5, markersize=3)
    axes[0].plot(estimates[:, 0], estimates[:, 1], 'b-', 
                 label='Kalman Estimate', linewidth=2.5, marker='s', markersize=2)
    axes[0].set_xlabel('X Position', fontsize=11)
    axes[0].set_ylabel('Y Position', fontsize=11)
    axes[0].set_title('Test 2: 2D Straight Line Tracking', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # Error magnitude over time
    axes[1].plot(measurement_error, 'r-', label='Measurement Error', alpha=0.7, linewidth=2)
    axes[1].plot(filter_error, 'b-', label='Filter Error', linewidth=2.5)
    axes[1].fill_between(range(num_steps), measurement_error, alpha=0.2, color='red')
    axes[1].fill_between(range(num_steps), filter_error, alpha=0.2, color='blue')
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Position Error (distance)', fontsize=11)
    axes[1].set_title('Error Over Time', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/test_2_2d_tracking.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: experiments/test_2_2d_tracking.png")

    return {
        'name': 'Test 2: 2D Straight Line',
        'measurement_rmse': measurement_rmse,
        'filter_rmse': filter_rmse,
        'improvement': improvement
    }


def test_2d_circular_motion():
    """
    Test 3: 2D Circular Motion (Model Mismatch)
    Tests filter performance on nonlinear motion
    """
    print("\n" + "="*60)
    print("Test 3: 2D Circular Motion (Model Mismatch)")
    print("="*60)

    dt = 0.1
    num_steps = 200
    radius = 10.0
    angular_velocity = 0.05

    # Real trajectory (circular motion)
    time = np.arange(num_steps) * dt
    true_positions = np.array([
        [radius * np.cos(angular_velocity * t), 
         radius * np.sin(angular_velocity * t)] 
        for t in time
    ])

    # Measurements with noise
    np.random.seed(44)
    noise = np.random.randn(num_steps, 2) * 0.2
    measurements = true_positions + noise

    # Initialize Kalman Filter with higher process noise (to adapt to nonlinearity)
    x0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    kf = KalmanFilter(dt=dt, x0=x0, Q=np.eye(4)*0.05)

    # Tracking loop
    estimates = []
    for i in range(num_steps):
        kf.predict()
        kf.update(measurements[i, :])
        estimates.append(kf.get_position())

    estimates = np.array(estimates)

    # Performance evaluation
    measurement_error = np.linalg.norm(measurements - true_positions, axis=1)
    filter_error = np.linalg.norm(estimates - true_positions, axis=1)

    measurement_rmse = np.sqrt(np.mean(measurement_error**2))
    filter_rmse = np.sqrt(np.mean(filter_error**2))
    improvement = ((measurement_rmse - filter_rmse) / measurement_rmse * 100) if measurement_rmse > 0 else 0

    print(f"Measurement RMSE: {measurement_rmse:.4f}")
    print(f"Filter RMSE:      {filter_rmse:.4f}")
    print(f"Improvement:      {improvement:.2f}%")
    print("⚠️  Note: Linear model has difficulty with nonlinear (circular) motion")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D Trajectory
    axes[0].plot(true_positions[:, 0], true_positions[:, 1], 'g-', 
                 label='True Circular Trajectory', linewidth=2.5, marker='o', markersize=2)
    axes[0].plot(measurements[:, 0], measurements[:, 1], 'r.', 
                 label='Measurements', alpha=0.3, markersize=2)
    axes[0].plot(estimates[:, 0], estimates[:, 1], 'b-', 
                 label='Kalman Estimate', linewidth=2.5, marker='s', markersize=2)
    axes[0].set_xlabel('X Position', fontsize=11)
    axes[0].set_ylabel('Y Position', fontsize=11)
    axes[0].set_title('Test 3: Circular Motion (Linear Model)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # Error over time
    axes[1].plot(measurement_error, 'r-', label='Measurement Error', alpha=0.7, linewidth=2)
    axes[1].plot(filter_error, 'b-', label='Filter Error', linewidth=2.5)
    axes[1].fill_between(range(num_steps), measurement_error, alpha=0.2, color='red')
    axes[1].fill_between(range(num_steps), filter_error, alpha=0.2, color='blue')
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Position Error (distance)', fontsize=11)
    axes[1].set_title('Error Over Time (Nonlinear Motion)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/test_3_circular_motion.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: experiments/test_3_circular_motion.png")

    return {
        'name': 'Test 3: Circular Motion',
        'measurement_rmse': measurement_rmse,
        'filter_rmse': filter_rmse,
        'improvement': improvement
    }


def test_noise_sensitivity():
    """
    Test 4: Noise Sensitivity Analysis
    Tests filter performance with different measurement noise levels
    """
    print("\n" + "="*60)
    print("Test 4: Noise Sensitivity Analysis")
    print("="*60)

    dt = 1.0
    num_steps = 100

    # Real trajectory (constant velocity)
    true_positions = np.arange(num_steps, dtype=np.float32)

    # Different noise levels to test
    noise_levels = [0.5, 1.0, 2.0, 4.0]
    results_detailed = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, noise_std in enumerate(noise_levels):
        # Generate measurements with noise
        np.random.seed(45 + idx)  # Different seed for each noise level
        noise = np.random.randn(num_steps) * noise_std
        measurements = true_positions + noise

        # Initialize filter with corresponding R value
        x0 = np.array([measurements[0], 0.0, 1.0, 0.0])
        R_matrix = np.eye(2) * (noise_std ** 2)
        kf = KalmanFilter(dt=dt, x0=x0, R=R_matrix)

        # Tracking loop
        estimates = []
        for i in range(num_steps):
            kf.predict()
            kf.update(np.array([measurements[i], 0.0]))
            est_pos = kf.get_position()
            estimates.append(est_pos[0])  # Extract x position

        estimates = np.array(estimates)

        # Error calculation
        measurement_error = measurements - true_positions
        filter_error = estimates - true_positions

        measurement_rmse = np.sqrt(np.mean(measurement_error**2))
        filter_rmse = np.sqrt(np.mean(filter_error**2))
        improvement = ((measurement_rmse - filter_rmse) / measurement_rmse * 100) if measurement_rmse > 0 else 0

        results_detailed.append({
            'noise_std': noise_std,
            'measurement_rmse': measurement_rmse,
            'filter_rmse': filter_rmse,
            'improvement': improvement
        })

        # Visualization
        axes[idx].plot(true_positions, 'g-', label='True Value', linewidth=2.5, marker='o', markersize=2)
        axes[idx].plot(measurements, 'r.', label='Measurements', alpha=0.5, markersize=3)
        axes[idx].plot(estimates, 'b-', label='Kalman Estimate', linewidth=2.5, marker='s', markersize=2)
        axes[idx].set_title(f'Noise Std = {noise_std:.1f}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Time Step', fontsize=10)
        axes[idx].set_ylabel('Position (x)', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/test_4_noise_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: experiments/test_4_noise_sensitivity.png")

    # Print results table
    print("\nNoise Sensitivity Analysis Results:")
    print("-" * 80)
    print(f"{'Noise Std':<12} {'Meas RMSE':<15} {'Filter RMSE':<15} {'Improvement (%)':<15}")
    print("-" * 80)
    for result in results_detailed:
        print(f"{result['noise_std']:<12.2f} {result['measurement_rmse']:<15.4f} "
              f"{result['filter_rmse']:<15.4f} {result['improvement']:<15.2f}")
    print("-" * 80)

    return {
        'name': 'Test 4: Noise Sensitivity',
        'results': results_detailed
    }


def main():
    """Run all Phase 1 tests"""
    ensure_output_dir()
    
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " " * 10 + "PHASE 1: Linear Kalman Filter Testing" + " " * 10 + "║")
    print("╚" + "="*58 + "╝")

    all_results = []

    # Test 1: 1D constant velocity
    result_1 = test_1d_constant_velocity()
    all_results.append(result_1)

    # Test 2: 2D straight line motion
    result_2 = test_2d_straight_line()
    all_results.append(result_2)

    # Test 3: 2D circular motion
    result_3 = test_2d_circular_motion()
    all_results.append(result_3)

    # Test 4: Noise sensitivity
    result_4 = test_noise_sensitivity()
    all_results.append(result_4)

    # Summary
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " " * 18 + "PHASE 1 COMPLETED!" + " " * 22 + "║")
    print("╚" + "="*58 + "╝")

    print("\n✅ Generated Test Results:")
    print("   • experiments/test_1_1d_tracking.png")
    print("   • experiments/test_2_2d_tracking.png")
    print("   • experiments/test_3_circular_motion.png")
    print("   • experiments/test_4_noise_sensitivity.png")

    print("\n📊 Summary of Improvements:")
    for result in all_results[:3]:  # First 3 tests
        print(f"   • {result['name']}: {result['improvement']:.1f}% improvement")

    print("\n🔄 Next Steps (Phase 2):")
    print("   1. Extract segmentation masks from UNet")
    print("   2. Calculate object centroids from masks")
    print("   3. Connect UNet output with Kalman Filter")
    print("   4. Real-time tracking on UAV video")

    print("\n🎯 Future Improvements (Phase 3+):")
    print("   • Extended Kalman Filter (EKF) for nonlinear models")
    print("   • Unscented Kalman Filter (UKF) for better estimate")
    print("   • Multi-object tracking")
    print("   • Parameter auto-tuning from data")

    print("\n✨ Phase 1 completed successfully!\n")


if __name__ == '__main__':
    main()
