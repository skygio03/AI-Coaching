"""Checks for the squat and bench press code using SIMULATED body points (no real video).

A stick figure with known angles is generated, a little noise and some missing frames are added,
and the analysis must find the same angles back and give (or not give) the expected tips.
This checks the maths and the rules. It does NOT check how well the pose model tracks a real
person: that needs real videos.

Run from the project folder:  python -m tests.test_gym_synthetic
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.common.pose_extractor import NAMES
from src.sports import bench_press, squat

FPS = 30.0
SIZE = 1000   # width and height of the blank test videos
RNG = np.random.default_rng(1)


def idx(name):
    return NAMES.index(name)


def add_noise(points, noise_px=1.5, missing=0.03):
    """Jitter every coordinate and blank some frames, like a real pose model does."""
    points = points + RNG.normal(0, noise_px, points.shape)
    gone = RNG.random(len(points)) < missing
    points[gone] = np.nan
    return points


# ------------------------------- squat ---------------------------------------

def squat_points(n_reps=5, rep_s=3.5, pause_s=1.0, flex=105, trunk=40, tibia=30,
                 heel_lift=0.0, facing=1, thigh=250.0, shank=250.0, trunk_len=300.0, foot=120.0):
    """Side-view stick figure of a lifter squatting; the angles are those at the lowest point."""
    n_pause, n_rep = int(pause_s * FPS), int(rep_s * FPS)
    progress = np.concatenate([np.zeros(n_pause)]
                              + [0.5 * (1 - np.cos(2 * np.pi * np.arange(n_rep) / n_rep))
                                 for _ in range(n_reps)]
                              + [np.zeros(n_pause)])
    points = np.full((len(progress), len(NAMES), 2), np.nan)
    ankle = np.array([500.0, 900.0])
    for i, p in enumerate(progress):
        f = 5 + (flex - 5) * p
        t = 5 + (trunk - 5) * p
        a = np.radians(3 + (tibia - 3) * p)
        knee = ankle + shank * np.array([np.sin(a), -np.cos(a)])
        s = np.array([-np.sin(a), np.cos(a)])           # unit vector knee -> ankle
        theta = np.radians(180 - f)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        hip = knee + thigh * (rot @ s)
        shoulder = hip + trunk_len * np.array([np.sin(np.radians(t)), -np.cos(np.radians(t))])
        heel = ankle + np.array([-0.35 * foot, 25.0 - heel_lift * p * foot])
        toe = ankle + np.array([0.65 * foot, 30.0])
        elbow, wrist = shoulder + [60, 120], shoulder + [150, 100]
        row = {"shoulder": shoulder, "elbow": elbow, "wrist": wrist, "hip": hip, "knee": knee,
               "ankle": ankle, "heel": heel, "toe": toe}
        for name, xy in row.items():
            points[i, idx(f"r_{name}")] = xy
            points[i, idx(f"l_{name}")] = xy + [8, 0]   # the other side, a little shifted
        points[i, idx("nose")] = shoulder + [40, -80]
    if facing == -1:
        points[..., 0] = SIZE - points[..., 0]
    return points


def blank_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    for _ in range(n_frames):
        writer.write(np.full((SIZE, SIZE, 3), 90, dtype=np.uint8))
    writer.release()


def squat_measures(points, side="r"):
    direction = squat.facing_direction(points, side)
    bottoms, y = squat.find_bottoms(points, FPS, side)
    starts = squat.find_starts(y, bottoms)
    return direction, bottoms, squat.analyze_reps(points, FPS, side, direction, bottoms, starts)


def titles(items):
    return [a.title for a in items]


def test_squat_finds_the_angles_back():
    for facing in (1, -1):
        points = add_noise(squat_points(facing=facing))
        direction, bottoms, m = squat_measures(points)
        assert direction == facing, (direction, facing)
        assert len(bottoms) == 5, f"found {len(bottoms)} squats instead of 5"
        for key, expected in (("flex", 105), ("trunk", 40), ("tibia", 30), ("diff", 10)):
            median = float(np.nanmedian(m[key]))
            assert abs(median - expected) < 3, f"{key}: {median:.1f} instead of {expected}"
    print("PASS squat: angles found back, both facing directions")


def test_squat_left_side_is_analysed_with_left_points():
    points = squat_points()
    # Break the right side: the left prefix must still work
    points[:, [idx(n) for n in NAMES if n.startswith("r_")]] = np.nan
    _, bottoms, m = squat_measures(add_noise(points), side="l")
    assert len(bottoms) == 5 and abs(float(np.nanmedian(m["flex"])) - 105) < 3
    print("PASS squat: left side analysed on its own")


def tips_for(**kwargs):
    points = add_noise(squat_points(**kwargs))
    _, _, m = squat_measures(points)
    return titles(squat.build_advice(m))


def test_squat_tips():
    neutral = tips_for(flex=105, trunk=32, tibia=30)
    assert "Your squat works mostly your hips" not in neutral
    assert "Your squat works mostly your thighs" not in neutral
    assert "Your heels come off the floor" not in neutral
    assert "You stop above thigh-level" not in neutral

    assert "Your squat works mostly your hips" in tips_for(trunk=50, tibia=25)
    assert "Your squat works mostly your thighs" in tips_for(trunk=20, tibia=40)
    assert "Your heels come off the floor" in tips_for(heel_lift=0.3)
    assert "You stop above thigh-level" in tips_for(flex=75)

    always = tips_for()
    for t in ("Knees in line with your toes", "Keep your back in its natural shape",
              "Feet about shoulder-width apart, toes turned out a little"):
        assert t in always
    print("PASS squat: tips appear only when they should")


def test_squat_unstable_squats_give_no_hip_knee_tip():
    # Alternate between a hip-heavy and a knee-heavy squat: no clear pattern, no advice
    a = squat_points(n_reps=1, trunk=55, tibia=20)
    b = squat_points(n_reps=1, trunk=15, tibia=45)
    points = add_noise(np.concatenate([a, b, a, b]))
    _, _, m = squat_measures(points)
    assert not {"Your squat works mostly your hips", "Your squat works mostly your thighs"} \
        & set(titles(squat.build_advice(m)))
    print("PASS squat: very different squats give no hips/knees advice")


# ------------------------------- bench press ---------------------------------

def bench_points(n_reps=5, rep_s=3.0, pause_s=1.0, ratio=1.5, tilt_deg=0.0, cx=500.0):
    """Foot-end stick figure: hands `ratio` times as far apart as the shoulders."""
    n_pause, n_rep = int(pause_s * FPS), int(rep_s * FPS)
    up = np.concatenate([np.ones(n_pause)]
                        + [0.5 * (1 + np.cos(2 * np.pi * np.arange(n_rep) / n_rep))
                           for _ in range(n_reps)]
                        + [np.ones(n_pause)])
    points = np.full((len(up), len(NAMES), 2), np.nan)
    sw, sy = 220.0, 500.0
    for i, p in enumerate(up):
        gap = ratio * sw
        wy = sy + 40 - 220 * p
        points[i, idx("l_shoulder")] = (cx - sw / 2, sy)
        points[i, idx("r_shoulder")] = (cx + sw / 2, sy)
        points[i, idx("l_wrist")] = (cx - gap / 2, wy)
        points[i, idx("r_wrist")] = (cx + gap / 2, wy + gap * np.tan(np.radians(tilt_deg)) * p)
        points[i, idx("l_elbow")] = (cx - gap / 2, sy + 60)
        points[i, idx("r_elbow")] = (cx + gap / 2, sy + 60)
        points[i, idx("l_hip")] = (cx - 80, 700)
        points[i, idx("r_hip")] = (cx + 80, 700)
    return points


def bench_measures(points):
    signal = bench_press.bar_signal(points, FPS)
    bottoms = bench_press.find_bottoms(signal, FPS)
    tops = bench_press.find_tops(signal, bottoms)
    return bottoms, bench_press.analyze_reps(points, FPS, bottoms, tops)


def bench_tips(**kwargs):
    _, m = bench_measures(add_noise(bench_points(**kwargs)))
    return titles(bench_press.build_advice(m))


def test_bench_finds_reps_and_ratio():
    bottoms, m = bench_measures(add_noise(bench_points(ratio=1.5)))
    assert len(bottoms) == 5, f"found {len(bottoms)} repetitions instead of 5"
    assert abs(float(np.nanmedian(m["grip"])) - 1.5) < 0.1
    assert abs(float(np.nanmedian(m["seconds"])) - 3.0) < 0.3
    print("PASS bench: repetitions, grip ratio and pace found back")


def test_bench_tips():
    assert "Your hands look wide on the bar" in bench_tips(ratio=2.0)
    assert "Your hands look narrow on the bar" in bench_tips(ratio=1.0)
    middle = bench_tips(ratio=1.5)
    assert "Your hands look wide on the bar" not in middle
    assert "Your hands look narrow on the bar" not in middle
    assert "One hand ends higher than the other" in bench_tips(tilt_deg=8)
    assert "One hand ends higher than the other" not in bench_tips(tilt_deg=0)
    assert "Shoulder blades back and down" in middle
    print("PASS bench: tips appear only when they should")


def test_line_angle_ignores_left_right_swap():
    left, right = np.array([[0.0, 0.0]]), np.array([[100.0, 10.0]])
    assert abs(bench_press.line_angle(left, right)[0] - bench_press.line_angle(right, left)[0]) < 1e-9
    print("PASS bench: line angle does not depend on which hand is called left")


# ------------------------------- whole analysis ------------------------------

def run_whole_analysis(module, points, **kwargs):
    """Run analyze_video with the pose model replaced by the simulated points."""
    original = module.extract_landmarks
    module.extract_landmarks = lambda path: (FPS, points)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            video = Path(tmp) / "blank.mp4"
            blank_video(video, len(points))
            result = module.analyze_video(video, output_dir=Path(tmp), **kwargs)
            assert Path(result["skeleton_path"]).is_file() and \
                Path(result["skeleton_path"]).stat().st_size > 0
            return result
    finally:
        module.extract_landmarks = original


def check_report(result, must_have):
    text = result["report_text"]
    for needle in must_have:
        assert needle in text, f"missing in the report: {needle!r}"
    assert {t for t, _ in result["sections"]}, "no sections"
    assert result["background"], "no background"
    assert "nan" not in text.lower().replace("finance", ""), "'nan' found in the report"


def test_whole_squat_analysis():
    points = add_noise(squat_points(trunk=50, tibia=25, heel_lift=0.3))
    result = run_whole_analysis(squat, points, recording="right")
    check_report(result, ["WHAT TO DO", "CHECK IT YOURSELF", "HOW MUCH TO TRUST THIS",
                          "Your heels come off the floor", "Your squat works mostly your hips",
                          "coach's tip", "backed by research", "check it yourself", "SOURCES"])
    # The first section must be the result and the advice must come before the numbers
    titles_ = [t for t, _ in result["sections"]]
    assert titles_[0] == "YOUR RESULT" and titles_[1] == "WHAT TO DO"
    try:
        run_whole_analysis(squat, points[:60], recording="right")
        raise AssertionError("a 2 s video should be refused")
    except ValueError:
        pass
    try:
        run_whole_analysis(squat, points, recording="front")
        raise AssertionError("an unknown side should be refused")
    except ValueError:
        pass
    print("PASS squat: whole analysis, report and refusals")


def test_whole_bench_analysis():
    points = add_noise(bench_points(ratio=2.0, tilt_deg=8))
    result = run_whole_analysis(bench_press, points, recording="foot_end")
    check_report(result, ["WHAT TO DO", "CHECK IT YOURSELF", "Your hands look wide on the bar",
                          "One hand ends higher than the other", "Shoulder blades back and down",
                          "SOURCES"])
    try:
        run_whole_analysis(bench_press, points, recording="side")
        raise AssertionError("an unknown camera position should be refused")
    except ValueError:
        pass
    # A video with no movement has no repetitions
    flat = bench_points(n_reps=0, pause_s=6.0)
    try:
        run_whole_analysis(bench_press, flat, recording="foot_end")
        raise AssertionError("a video with no repetitions should be refused")
    except ValueError:
        pass
    print("PASS bench: whole analysis, report and refusals")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} checks passed.")


if __name__ == "__main__":
    main()