import numpy as np


def compute_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute angle in degrees between two sets of 3D vectors."""
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
    dot = np.sum((v1 / norm1) * (v2 / norm2), axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def extract_clinical_gait_features(joints: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """
    Extract 30+ bio-mechanical gait features from 3D joint positions (T, 22, 3).
    Joint indices:
      0: Root/Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee,
      6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3/Chest, 10: L_Foot, 11: R_Foot,
      12: Neck, 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow,
      20: L_Wrist, 21: R_Wrist
    """
    features = []
    T = joints.shape[0]
    dt = 1.0 / max(fps, 1.0)

    # 1. Arm Swing & Upper Body Asymmetry
    # Left Arm (Wrist 20 relative to Hip 1), Right Arm (Wrist 21 relative to Hip 2)
    l_arm_disp = joints[:, 20, :] - joints[:, 1, :]
    r_arm_disp = joints[:, 21, :] - joints[:, 2, :]

    l_arm_rom = np.ptp(l_arm_disp, axis=0)  # [range_x, range_y, range_z]
    r_arm_rom = np.ptp(r_arm_disp, axis=0)

    l_arm_mag = np.linalg.norm(l_arm_rom)
    r_arm_mag = np.linalg.norm(r_arm_rom)

    arm_asymmetry_abs = abs(l_arm_mag - r_arm_mag)
    arm_asymmetry_ratio = arm_asymmetry_abs / (l_arm_mag + r_arm_mag + 1e-5)

    features.extend([l_arm_mag, r_arm_mag, arm_asymmetry_abs, arm_asymmetry_ratio])
    features.extend(l_arm_rom)
    features.extend(r_arm_rom)

    # 2. Leg Velocity & Swing Asymmetry
    l_ankle_v = np.diff(joints[:, 7, :], axis=0) / dt
    r_ankle_v = np.diff(joints[:, 8, :], axis=0) / dt

    l_ankle_speed = np.linalg.norm(l_ankle_v, axis=1)
    r_ankle_speed = np.linalg.norm(r_ankle_v, axis=1)

    l_speed_max, l_speed_mean, l_speed_std = np.max(l_ankle_speed), np.mean(l_ankle_speed), np.std(l_ankle_speed)
    r_speed_max, r_speed_mean, r_speed_std = np.max(r_ankle_speed), np.mean(r_ankle_speed), np.std(r_ankle_speed)

    leg_asymmetry_max = abs(l_speed_max - r_speed_max) / (l_speed_max + r_speed_max + 1e-5)
    leg_asymmetry_mean = abs(l_speed_mean - r_speed_mean) / (l_speed_mean + r_speed_mean + 1e-5)

    features.extend([l_speed_max, l_speed_mean, l_speed_std,
                     r_speed_max, r_speed_mean, r_speed_std,
                     leg_asymmetry_max, leg_asymmetry_mean])

    # 3. Knee Flexion / Extension Angles (Range of Motion)
    # L_Knee: 4 (between 1-4 and 7-4), R_Knee: 5 (between 2-5 and 8-5)
    v1_l = joints[:, 1, :] - joints[:, 4, :]
    v2_l = joints[:, 7, :] - joints[:, 4, :]
    l_knee_angles = compute_angle(v1_l, v2_l)

    v1_r = joints[:, 2, :] - joints[:, 5, :]
    v2_r = joints[:, 8, :] - joints[:, 5, :]
    r_knee_angles = compute_angle(v1_r, v2_r)

    l_knee_rom = np.ptp(l_knee_angles)
    r_knee_rom = np.ptp(r_knee_angles)
    knee_asymmetry = abs(l_knee_rom - r_knee_rom)

    features.extend([np.mean(l_knee_angles), np.std(l_knee_angles), l_knee_rom,
                     np.mean(r_knee_angles), np.std(r_knee_angles), r_knee_rom,
                     knee_asymmetry])

    # 4. Postural Stability & Trunk Tilt
    # Trunk vector from Pelvis (0) to Spine3/Chest (9)
    trunk_vec = joints[:, 9, :] - joints[:, 0, :]
    trunk_len = np.linalg.norm(trunk_vec, axis=1, keepdims=True) + 1e-8
    trunk_dir = trunk_vec / trunk_len

    # Vertical inclination (angle with Y-axis)
    vert_axis = np.array([0.0, 1.0, 0.0])
    trunk_tilt = compute_angle(trunk_dir, vert_axis)

    pelvis_lateral_var = np.var(joints[:, 0, 0])  # Pelvis X sway
    pelvis_vert_var = np.var(joints[:, 0, 1])     # Pelvis Y bounce

    features.extend([np.mean(trunk_tilt), np.std(trunk_tilt), np.max(trunk_tilt),
                     pelvis_lateral_var, pelvis_vert_var])

    # 5. Frequency Domain: Tremor & Shuffling Power Ratio (3-8 Hz vs 0.5-2.0 Hz)
    # FFT on wrist & ankle accelerations
    l_wrist_acc = np.diff(joints[:, 20, :], n=2, axis=0) / (dt**2)
    l_wrist_acc_mag = np.linalg.norm(l_wrist_acc, axis=1)

    fft_vals = np.abs(np.fft.rfft(l_wrist_acc_mag))
    fft_freqs = np.fft.rfftfreq(len(l_wrist_acc_mag), d=dt)

    gait_mask = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
    tremor_mask = (fft_freqs >= 3.0) & (fft_freqs <= 8.0)

    gait_power = np.sum(fft_vals[gait_mask]**2) if np.any(gait_mask) else 0.0
    tremor_power = np.sum(fft_vals[tremor_mask]**2) if np.any(tremor_mask) else 0.0

    tremor_ratio = tremor_power / (gait_power + 1e-5)
    features.extend([gait_power, tremor_power, tremor_ratio])

    # 6. Global Gait Speed & Cadence
    root_vel = np.diff(joints[:, 0, :], axis=0) / dt
    root_speed = np.linalg.norm(root_vel, axis=1)

    features.extend([np.mean(root_speed), np.std(root_speed), np.max(root_speed)])

    return np.array(features, dtype=np.float32)
