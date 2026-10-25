"""Analyze a running video: cadence, foot strikes, contact time and knee angle."""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.pose_extractor import extract_landmarks, NAMES
from src.render import render_skeleton


def joint_angle(a, b, c):
    """Angle in degrees at point b, formed by the segments b->a and b->c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def fill_gaps(y):
    """Replace NaN values (frames with no detection) by linear interpolation."""
    valid = ~np.isnan(y)
    return np.interp(np.arange(len(y)), np.flatnonzero(valid), y[valid])


def smooth(y, fps):
    """Savitzky-Golay filter: removes jitter but keeps the shape of the curve."""
    # Window of about 0.1 s, forced to be odd and at least 5 samples
    window = max(int(0.1 * fps) | 1, 5)
    return savgol_filter(y, window, 2)


def foot_height_signal(points, fps, foot):
    """Vertical position of the lowest point of the foot ('l' or 'r').

    In image coordinates y grows downward, so a larger value means a lower foot.
    """
    heel = points[:, NAMES.index(f"{foot}_heel"), 1]
    toe = points[:, NAMES.index(f"{foot}_toe"), 1]
    # The lowest point of the foot is the max y; works for heel and forefoot strikers
    lowest = np.fmax(heel, toe)
    return smooth(fill_gaps(lowest), fps)


def detect_strikes(points, fps, foot, level=0.83, window_s=0.3, ground_window_s=1.0):
    """Return (touchdowns, toe_offs) as arrays of frame indices for one foot.

    The foot height has a plateau while the foot is on the ground. The plateau height
    ("ground") is estimated locally around each strike, so it follows the runner if
    they drift in the frame (or the camera tilts) during the video. Starting from the
    middle of each plateau, walk outwards while the foot stays near the ground:
    - touchdown = first frame of the run where the foot is at `level` of the way
      between its highest point in the window and the ground height
    - toe-off = last frame of that run
    Lower `level` = touchdown detected earlier, higher `level` = later.
    """
    y = foot_height_signal(points, fps, foot)

    # Two strikes of the same foot cannot be closer than ~0.5 s
    peaks, _ = find_peaks(y, distance=int(0.5 * fps))

    touchdowns, toe_offs = [], []
    for p in peaks:
        start = max(0, p - int(window_s * fps))
        end = min(len(y) - 1, p + int(window_s * fps))
        top = y[start:end + 1].min()              # highest position of the foot

        # Local ground height: 80th percentile of the foot height within +/- 1 s.
        # The foot is on the ground about a third of the time, so this lies on the
        # stance plateau and is not moved by isolated spikes in the signal
        lo = max(0, p - int(ground_window_s * fps))
        hi = min(len(y), p + int(ground_window_s * fps) + 1)
        ground = np.percentile(y[lo:hi], 80)

        # The ground can never be above the peak itself, otherwise the threshold
        # would sit under the foot and the stance window would collapse to 0 ms
        ground = min(ground, y[p])
        threshold = top + level * (ground - top)  # almost at ground height

        # Stop at the first frame that leaves the plateau, so a later spike in the
        # signal (e.g. a keypoint swapped with the other leg) cannot stretch the stance
        a = p
        while a > start and y[a - 1] >= threshold:
            a -= 1
        b = p
        while b < end and y[b + 1] >= threshold:
            b += 1

        touchdowns.append(a)
        toe_offs.append(b)

    return np.array(touchdowns, dtype=int), np.array(toe_offs, dtype=int)

def cadence_from_strides(touchdowns_by_foot, fps):
    """Return (cadence in steps/min, strides used, strides discarded).

    Uses same-foot intervals (one stride = two steps), so a small left/right
    timing offset in the detection does not bias the result.
    """
    intervals = [np.diff(td) / fps for td in touchdowns_by_foot if len(td) > 1]
    if not intervals:
        return float("nan"), 0, 0
    strides = np.concatenate(intervals)

    # A missed strike makes a stride about twice as long; keep only plausible values
    ok = (strides >= 0.5) & (strides <= 1.0)
    if not ok.any():
        return float("nan"), 0, len(strides)
    return 120.0 / float(np.median(strides[ok])), int(ok.sum()), int((~ok).sum())


def knee_angle_at(points, f, foot):
    """Hip-knee-ankle angle at frame f (180 = straight leg); NaN if a landmark is missing."""
    hip, knee, ankle = (points[f, NAMES.index(f"{foot}_{n}")]
                        for n in ("hip", "knee", "ankle"))
    if np.isnan([hip, knee, ankle]).any():
        return float("nan")
    return joint_angle(hip, knee, ankle)


def summarize(values):
    """Return (median, 25th percentile, 75th percentile, count), ignoring NaN.

    The 25th-75th range (middle half of the strikes) is robust to a few wrong
    detections, unlike the standard deviation.
    """
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), len(v))


def report(label, values, unit):
    """Print one measurement as: median (middle half of strikes: lo-hi, n strikes)."""
    med, lo, hi, n = summarize(values)
    print(f"{label}: {med:.0f} {unit} (middle half of strikes: {lo:.0f}-{hi:.0f}, {n} strikes)")


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input/run.mp4")
    if not video.is_file():
        print(f"Video not found: {video}")
        return

    print("Extracting landmarks (this can take a minute)...")
    fps, points = extract_landmarks(video)
    duration = len(points) / fps

    left_td, left_off = detect_strikes(points, fps, "l")
    right_td, right_off = detect_strikes(points, fps, "r")

    print(f"Video: {duration:.1f}s at {fps:.0f} fps")
    print(f"Foot strikes: {len(left_td)} left, {len(right_td)} right")

    cadence, used, discarded = cadence_from_strides([left_td, right_td], fps)
    if used:
        print(f"Cadence: {cadence:.0f} steps/min  (from {used} strides, {discarded} discarded)")
    else:
        print("Too few foot strikes detected to compute cadence.")

        # Per-leg measurements
    for foot, label, td, off in (("l", "Left", left_td, left_off),
                                 ("r", "Right", right_td, right_off)):
        # Ground contact time in ms; values outside 100-500 ms are detection errors
        contact = np.array([(b - a) / fps * 1000 for a, b in zip(td, off)])
        plausible = (contact >= 100) & (contact <= 500)

                # List the discarded strikes (time in seconds) so they can be checked in the video
        if (~plausible).any():
            bad = [f"{a / fps:.1f}s ({(b - a) / fps * 1000:.0f} ms)"
                   for a, b, ok in zip(td, off, plausible) if not ok]
            print(f"  {label} discarded strikes at: {', '.join(bad)}")

        # Keep only strikes with a plausible stance window, so swing-phase frames
        # (where the knee is very bent) never leak into the stance measurements
        td_ok, off_ok = td[plausible], off[plausible]

        at_touchdown = [knee_angle_at(points, f, foot) for f in td_ok]

        # Peak flexion = smallest knee angle between touchdown and toe-off
        peak_flexion = []
        for a, b in zip(td_ok, off_ok):
            angles = [knee_angle_at(points, f, foot) for f in range(a, b + 1)]
            angles = [x for x in angles if not np.isnan(x)]
            peak_flexion.append(min(angles) if angles else float("nan"))

        report(f"{label} knee at touchdown", at_touchdown, "deg")
        report(f"{label} knee peak flexion in stance", peak_flexion, "deg")
        report(f"{label} contact time", contact[plausible], "ms")
        if (~plausible).any():
            print(f"  ({int((~plausible).sum())} of {len(td)} strikes discarded: implausible stance window)")

    # Save a copy of the video with the skeleton and the touchdown frames marked
    out_path = Path("data/output") / f"{video.stem}_skeleton.mp4"
    render_skeleton(video, out_path, fps, points, left_td, right_td)
    print(f"Skeleton video saved to: {out_path}")


if __name__ == "__main__":
    main()