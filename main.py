"""Analyze a running video: cadence and knee angle at foot strike."""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from src.pose_extractor import extract_landmarks, NAMES


def joint_angle(a, b, c):
    """Angle in degrees at point b, formed by the segments b->a and b->c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def foot_strikes(points, fps, foot):
    """Return frame indices of foot strikes for one foot ('l' or 'r').

    First approximation: a strike is a local maximum of the ankle's y value
    (in image coordinates y grows downward, so max y = lowest point).
    """
    y = points[:, NAMES.index(f"{foot}_ankle"), 1]

    # Fill NaN gaps by interpolation so the peak finder does not break
    valid = ~np.isnan(y)
    y = np.interp(np.arange(len(y)), np.flatnonzero(valid), y[valid])

    # Two strikes of the same foot cannot be closer than ~0.4 s
    peaks, _ = find_peaks(y, distance=int(0.4 * fps))
    return peaks


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input/run.mp4")
    if not video.is_file():
        print(f"Video not found: {video}")
        return

    print("Extracting landmarks (this can take a minute)...")
    fps, points = extract_landmarks(video)
    duration = len(points) / fps

    left = foot_strikes(points, fps, "l")
    right = foot_strikes(points, fps, "r")
    cadence = (len(left) + len(right)) / duration * 60

    print(f"Video: {duration:.1f}s at {fps:.0f} fps")
    print(f"Foot strikes: {len(left)} left, {len(right)} right")
    print(f"Cadence: {cadence:.0f} steps/min  (typical recreational range: 160-190)")

    # Knee angle at each left foot strike (180 = fully straight leg)
    angles = []
    for f in left:
        hip, knee, ankle = (points[f, NAMES.index(n)] for n in ("l_hip", "l_knee", "l_ankle"))
        if not np.isnan([hip, knee, ankle]).any():
            angles.append(joint_angle(hip, knee, ankle))
    if angles:
        print(f"Left knee angle at strike: {np.mean(angles):.0f} degrees on average")


# This block actually starts the program; without it, running the file prints nothing
if __name__ == "__main__":
    main()