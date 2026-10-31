"""Squat analysis from a SIDE view: depth, chest and shin lean, heels, with plain-language tips.

The camera is beside the lifter, so that one side of the body faces it (left or right).
Only that side is analysed and drawn.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from src.common.advice import (Advice, CHECK, COACH, RESEARCH, check_section, extra_section,
                               source_lines, what_to_do_section)
from src.common.pose_extractor import NAMES, extract_landmarks
from src.common.render import render_skeleton, to_h264
# Helpers shared with the running analysis
from src.sports.running import fill_gaps, fmt, report_to_text, smooth, summarize

# ---------------------------------------------------------------------------
# Settings. Values marked "assumption" are my own choices, NOT taken from the
# papers; check them against your videos.
# ---------------------------------------------------------------------------
MIN_REPS = 3                # assumption: fewer squats give too little data
MIN_DURATION_S = 5.0        # assumption
MIN_DETECTED = 0.6          # assumption: share of frames in which the leg must be found
MIN_HIP_DROP = 0.20         # assumption: the hip must go down by at least 20% of the leg length
MIN_REP_TIME_S = 1.0        # assumption: two squats cannot be closer than this
MIN_KNEE_BEND_DEG = 40      # assumption: a smaller knee bend is not counted as a squat
WINDOW_S = 0.07             # assumption: angles are read around the lowest point (median)
MAX_SPREAD_DEG = 20         # assumption: if the middle half of the squats is wider, no advice
HEEL_LIFT_FRACTION = 0.15   # assumption: heel rise (in foot lengths) that counts as "heels up"

# From Straub & Powers 2024 (their working definitions, not strict limits):
NEUTRAL_BAND_DEG = 10       # chest lean minus shin lean within +-10 deg: hips and knees share the work
DEPTH_PARTIAL_DEG = 90      # knee bend below this: partial squat (above thigh-level)
DEPTH_DEEP_DEG = 110        # knee bend above this: deep squat (below thigh-level)

SIDES = {"left": "l", "right": "r"}

# Interface
RECORDING_OPTIONS = {
    "My left side faces the camera": "left",
    "My right side faces the camera": "right",
}
OPTION_QUESTION = "Side of your body facing the camera"
VIDEO_NOTE = ("Only the side facing the camera is drawn. A ring marks the hip at the lowest "
              "point of each squat.")
NOTICE = ("The squat analysis is new and has not been tested on real videos yet: check that "
          "the skeleton follows your body and treat the numbers with caution.")


# ------------------------------- helpers -------------------------------------

def nanmed(values):
    """Median ignoring NaN; NaN if there is nothing to take the median of."""
    a = np.asarray(values, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmedian(a))


def landmark(points, prefix, name):
    """Pixel positions of one landmark of the analysed side, shape [n_frames, 2]."""
    return points[:, NAMES.index(f"{prefix}_{name}")]


def angle_series(a, b, c):
    """Interior angle at b (degrees) for every frame; NaN where a point is missing."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def lean_series(top, bottom, direction):
    """Angle of the line bottom->top from the vertical, positive when it leans forward.

    `direction` is +1 if the lifter faces the right of the image, -1 if the left.
    Image y grows downwards, so "up" is -y.
    """
    v = top - bottom
    return np.degrees(np.arctan2(direction * v[:, 0], -v[:, 1]))


def facing_direction(points, prefix):
    """+1 if the lifter faces the right of the image, -1 if the left (toes are in front of heels)."""
    dx = nanmed(landmark(points, prefix, "toe")[:, 0] - landmark(points, prefix, "heel")[:, 0])
    if not np.isfinite(dx) or abs(dx) < 1e-6:
        raise ValueError("Could not tell which way you are facing. Make sure your whole body, "
                         "feet included, is visible from the side.")
    return 1 if dx > 0 else -1


def leg_length_px(points, prefix):
    """Median thigh + shank length of the analysed leg, in pixels."""
    hip, knee, ankle = (landmark(points, prefix, n) for n in ("hip", "knee", "ankle"))
    return nanmed(np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1))


def find_bottoms(points, fps, prefix):
    """Frames of the lowest point of each squat (the hip is at its lowest), and the hip signal."""
    y = smooth(fill_gaps(landmark(points, prefix, "hip")[:, 1]), fps)
    leg = leg_length_px(points, prefix)
    # Image y grows downwards, so the lowest hip position is a peak of y
    peaks, _ = find_peaks(y, prominence=MIN_HIP_DROP * leg,
                          distance=max(1, int(MIN_REP_TIME_S * fps)))
    return peaks, y


def find_starts(y, bottoms):
    """For each squat, the frame where the hip was highest before it (standing)."""
    starts, previous = [], 0
    for b in bottoms:
        starts.append(previous + int(np.argmin(y[previous:b + 1])))
        previous = b
    return starts


def analyze_reps(points, fps, prefix, direction, bottoms, starts):
    """Measure every squat at its lowest point. Returns a dict of lists (one value per squat)."""
    hip, knee, ankle, shoulder, heel, toe = (
        landmark(points, prefix, n) for n in ("hip", "knee", "ankle", "shoulder", "heel", "toe"))
    flex = 180.0 - angle_series(hip, knee, ankle)       # 0 deg = straight leg
    trunk = lean_series(shoulder, hip, direction)       # chest lean from upright
    tibia = lean_series(knee, ankle, direction)         # shin lean from upright
    foot_len = nanmed(np.linalg.norm(heel - toe, axis=1))
    w = max(1, int(round(WINDOW_S * fps)))

    out = {"flex": [], "trunk": [], "tibia": [], "diff": [], "heel": [], "descent_s": []}
    for s, b in zip(starts, bottoms):
        low = slice(max(b - w, 0), b + w + 1)
        stand = slice(max(s - w, 0), s + w + 1)
        f = nanmed(flex[low])
        if not np.isfinite(f) or f < MIN_KNEE_BEND_DEG:
            continue
        t, ti = nanmed(trunk[low]), nanmed(tibia[low])
        out["flex"].append(f)
        out["trunk"].append(t)
        out["tibia"].append(ti)
        out["diff"].append(t - ti)
        # Heel rise measured against the toe, so that the whole foot sliding is not counted
        rise = ((nanmed(heel[stand, 1]) - nanmed(heel[low, 1]))
                - (nanmed(toe[stand, 1]) - nanmed(toe[low, 1]))) / foot_len
        out["heel"].append(rise)
        out["descent_s"].append((b - s) / fps)
    return out


# ------------------------------- advice --------------------------------------

def num(value, decimals=0):
    """Format a number for the report, or 'n/a' if it could not be measured."""
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


def depth_word(flex):
    """Plain words for the knee bend at the bottom."""
    if flex < DEPTH_PARTIAL_DEG:
        return "above thigh-level"
    if flex <= DEPTH_DEEP_DEG:
        return "about thigh-level"
    return "below thigh-level, a deep squat"


def stable(values, limit=None):
    """Return (median, ok): ok if there are enough squats and they are similar to each other."""
    med, lo, hi, n = summarize(values)
    ok = n >= MIN_REPS and (limit is None or (hi - lo) <= limit)
    return med, ok


def check_items():
    """Things a side view cannot show: the person has to check them alone."""
    return [
        Advice(
            CHECK, "Knees in line with your toes",
            try_this="On the way down and on the way up, keep your knees pointing the same way "
                     "as your toes. Do not let them drift inwards. Film yourself from the front "
                     "or use a mirror to check.",
            why="Knees falling inwards during the squat are described in studies as a risk factor "
                "for knee injuries.",
            source="Lorenzetti et al. 2018 (introduction and discussion: general guideline that "
                   "the knees stay between the ankles).",
        ),
        Advice(
            CHECK, "Feet about shoulder-width apart, toes turned out a little",
            try_this="Stand about as wide as your shoulders (a bit wider is fine), with your toes "
                     "turned out slightly, about 20 degrees.",
            why="In a study of 42 people the authors recommend this set-up. A very narrow stance "
                "with the feet turned out a lot, or a very wide stance with the feet straight, "
                "gave the largest loads at knees and hips.",
            source="Lorenzetti et al. 2018; Straub & Powers 2024 (turning the toes out 30 degrees "
                   "cut the inward knee load by about half in one study they cite).",
        ),
        Advice(
            CHECK, "Keep your back in its natural shape",
            try_this="Watch for the moment your lower back starts to round at the bottom. If it "
                     "does, stop a little higher.",
            why="When you go lower than your hips can bend, your pelvis tips under and your lower "
                "back rounds, which puts more stress on the spine.",
            source="Straub & Powers 2024 (posterior pelvic tilt with lumbar flexion at the end of "
                   "the hip range).",
        ),
    ]


def build_advice(measures):
    """Turn the measures into tips. Only tips supported by enough similar squats are given."""
    items = []

    heel, heel_ok = stable(measures["heel"])
    if heel_ok and heel >= HEEL_LIFT_FRACTION:
        items.append(Advice(
            COACH, "Your heels come off the floor",
            saw=f"At the bottom your heels rise by about {heel * 100:.0f}% of the length of your foot.",
            try_this="Keep your whole foot flat on the floor and push through the middle of your "
                     "foot. If your heels keep rising, try a slightly wider stance and check "
                     "whether your ankles are stiff.",
            why="Raised heels let your knees travel further forward, which puts more of the work "
                "on the front of your thighs and knees.",
            sure="low", priority=1,
            source="Common coaching advice. Straub & Powers 2024 only show that raised heels "
                   "increase the forward lean of the shin and the load at the knee.",
        ))

    diff, diff_ok = stable(measures["diff"], MAX_SPREAD_DEG)
    trunk, _ = stable(measures["trunk"])
    tibia, _ = stable(measures["tibia"])
    if diff_ok and diff > NEUTRAL_BAND_DEG:
        items.append(Advice(
            RESEARCH, "Your squat works mostly your hips",
            saw=f"At the bottom your chest leans about {trunk:.0f} degrees from upright and your "
                f"shins about {tibia:.0f}: the chest leans {diff:.0f} degrees more.",
            try_this="This is fine if you want to train hips and glutes. If you want more work "
                     "for the front of your thighs, stay a bit more upright and let your knees "
                     "move forward over your toes.",
            why="The more your chest leans compared with your shins, the more your hips work and "
                "the less your knees.",
            sure="medium", priority=2,
            source="Straub & Powers 2024 (review; based on Barrack et al.): a difference above "
                   "10 degrees means the hips do more of the work.",
        ))
    elif diff_ok and diff < -NEUTRAL_BAND_DEG:
        items.append(Advice(
            RESEARCH, "Your squat works mostly your thighs",
            saw=f"At the bottom your chest leans about {trunk:.0f} degrees from upright and your "
                f"shins about {tibia:.0f}: the shins lean {-diff:.0f} degrees more.",
            try_this="This is fine if you want strong thighs. If you want more work for hips and "
                     "glutes, push your hips back more and let your chest lean a little more.",
            why="The more your shins lean compared with your chest, the more your knees work and "
                "the less your hips.",
            sure="medium", priority=2,
            source="Straub & Powers 2024 (review; based on Barrack et al.): a difference below "
                   "-10 degrees means the knees do more of the work.",
        ))

    flex, flex_ok = stable(measures["flex"], MAX_SPREAD_DEG)
    if flex_ok and flex < DEPTH_PARTIAL_DEG:
        items.append(Advice(
            COACH, "You stop above thigh-level",
            saw=f"At the bottom your knees are bent about {flex:.0f} degrees.",
            try_this="If your knees and hips feel fine, many coaches suggest going down until "
                     "your thighs are about level with the floor (roughly 90 to 110 degrees of "
                     "knee bend). If a deeper squat hurts your knees or you cannot keep your "
                     "back in shape, stay where you are.",
            why="Going deeper increases the load on hips and knees, so how deep is right depends "
                "on your goal and on how flexible your hips are.",
            sure="low", priority=3,
            source="Common coaching advice. Straub & Powers 2024 report that depth findings "
                   "conflict and that hip mobility limits it.",
        ))

    return items + check_items()


# ------------------------------- the report ----------------------------------

def filming_tips(recording="right"):
    """Return the filming tips for the squat, as a list of '- ' lines."""
    return [
        "- Camera beside you, at about hip height, on a tripod, 2.5 to 3 metres away, looking "
        "straight at your side (not from the front or at an angle).",
        "- Choose below which side of your body faces the camera: only that side is analysed.",
        "- Your whole body must stay in the picture, from head to feet, including the bar. Do not "
        "cut off your feet.",
        "- Tight-fitting clothes and good light. A bar on your shoulders can hide them: check that "
        "the skeleton still follows your body.",
        f"- Film a whole set: at least {MIN_REPS} squats (5 or more is better), without moving "
        "the camera.",
        "- 60 frames per second or more if your phone allows it.",
        "- These camera positions are suggestions of this project, not values from the studies.",
    ]


def build_report(fps, duration, reps, measures, items):
    """Return (main_sections, background_sections) as lists of (title, [lines])."""
    flex, _, _, n = summarize(measures["flex"])
    diff, diff_ok = stable(measures["diff"], MAX_SPREAD_DEG)
    trunk, _, _, _ = summarize(measures["trunk"])
    tibia, _, _, _ = summarize(measures["tibia"])
    heel, _, _, _ = summarize(measures["heel"])

    main = [("YOUR RESULT", [
        f"Video: {duration:.1f} s at {fps:.0f} fps, {reps} squats counted.",
        "These are estimates from a single 2D video: not a medical opinion and not a coach.",
    ])]
    main.append(what_to_do_section(
        items, "Nothing stands out in what I can measure from the side. Still go through the "
               "list below, because the camera cannot see everything."))

    measured = [f"- Squats counted: {reps}."]
    measured.append(f"- Depth: at the bottom your knees are bent about {flex:.0f} degrees "
                    f"({depth_word(flex)}).")
    measured.append(f"- Chest lean at the bottom: about {num(trunk)} degrees from upright.")
    measured.append(f"- Shin lean at the bottom: about {num(tibia)} degrees from upright "
                    "(how far your knees travel forward over your feet).")
    if diff_ok:
        if diff > NEUTRAL_BAND_DEG:
            share = "your chest leans more than your shins, so your hips do more of the work"
        elif diff < -NEUTRAL_BAND_DEG:
            share = "your shins lean more than your chest, so your knees do more of the work"
        else:
            share = "chest and shins lean by a similar amount, so hips and knees share the work"
        measured.append(f"- Hips or knees: {share}.")
    if np.isfinite(heel):
        measured.append("- Heels: they stay on the floor." if heel < HEEL_LIFT_FRACTION
                        else f"- Heels: they rise by about {heel * 100:.0f}% of your foot length.")
    main.append(("WHAT I MEASURED", measured))
    main.append(check_section(items))

    trust = [f"- {reps} squats were analysed (at least {MIN_REPS} are needed).",
             "- Angles measured from a video can be off by several degrees, more if the camera is "
             "not exactly beside you or not at hip height."]
    if reps >= MIN_REPS and not diff_ok:
        trust.append("- Your squats differ a lot from one to the next, so I did not give advice "
                     "about hips and knees.")
    trust.append("- Only the side facing the camera is analysed, so left and right differences "
                 "cannot be seen.")
    main.append(("HOW MUCH TO TRUST THIS", trust))

    bg = []
    extra = extra_section(items)
    if extra:
        bg.append(extra)
    bg.append(("THE NUMBERS", [
        f"- Knee bend at the bottom (0 = straight leg): {fmt(measures['flex'])} degrees",
        f"- Chest lean from upright at the bottom: {fmt(measures['trunk'])} degrees",
        f"- Shin lean from upright at the bottom: {fmt(measures['tibia'])} degrees",
        f"- Chest lean minus shin lean: {fmt(measures['diff'])} degrees",
        f"- Heel rise as a share of foot length: {fmt([h * 100 for h in measures['heel']])} %",
        f"- Time from standing to the bottom: {fmt(measures['descent_s'], 1)} s",
        "  'from X to Y' is the middle half of the squats, i.e. how much the value changes "
        "from one squat to the next.",
    ]))
    bg.append(("HOW THE TIPS WERE DECIDED", [
        f"- Hips or knees: the chest lean minus the shin lean at the lowest point. Above "
        f"{NEUTRAL_BAND_DEG} degrees the hips do more of the work, below -{NEUTRAL_BAND_DEG} the "
        "knees do. The limit comes from Straub & Powers (2024), who base it on Barrack et al.; "
        "it is a working definition, not a measured threshold.",
        f"- Depth: below {DEPTH_PARTIAL_DEG} degrees of knee bend is called partial, "
        f"{DEPTH_PARTIAL_DEG}-{DEPTH_DEEP_DEG} about thigh-level, above {DEPTH_DEEP_DEG} deep "
        "(Straub & Powers 2024).",
        f"- Heels: counted as raised when the heel rises by {HEEL_LIFT_FRACTION:.0%} of the foot "
        "length or more. This limit is an assumption of this project.",
        f"- A tip is given only if at least {MIN_REPS} squats were found and, for hips and "
        f"knees and for depth, the middle half of the squats is within {MAX_SPREAD_DEG} degrees. "
        "This limit is an assumption of this project.",
        "- The squats are found from the movement of the hip: a squat counts if the hip goes "
        f"down by at least {MIN_HIP_DROP:.0%} of the leg length and the knee bends at least "
        f"{MIN_KNEE_BEND_DEG} degrees (assumptions).",
        "- Angles are read as the median of a few frames around the lowest point.",
    ]))
    bg.append(("WHAT CANNOT BE MEASURED FROM THIS VIDEO", [
        "- Whether the knees drift inwards or outwards (a front view would be needed).",
        "- Width of the stance and the angle of the feet.",
        "- Rounding of the lower back (the model only joins shoulder and hip).",
        "- The weight on the bar, the position of the bar and the speed of the lift.",
        "- The side that is turned away from the camera.",
    ]))
    bg.append(("WHAT THE RESEARCH SAYS", [
        "- A 2024 review (Straub & Powers; a descriptive review, the lowest level of evidence on "
        "their scale) explains that the more the chest leans compared with the shins, the more the "
        "hips work and the less the knees; it uses a difference of about 10 degrees to separate "
        "hip-focused, balanced and knee-focused squats.",
        "- The same review reports that turning the toes out by 30 degrees lowered the inward "
        "knee load by about half in one study, that a stance wider than 150% of the shoulders "
        "raised it, and that turning the hips out 30-50 degrees increased the activity of the "
        "inner-thigh muscles (adductors) compared with feet straight. It does not say that "
        "knees drifting inwards 'switch on' the adductors.",
        "- Findings about depth conflict: some studies find more muscle activity in the thigh and "
        "buttock muscles with depth and some do not. The limit to depth is the bend available at "
        "the hip; beyond it the pelvis tips under and the lower back rounds.",
        "- Raising the heels lets the shin lean further forward and raises the load at the knee.",
        "- A 2018 study (Lorenzetti et al.; 21 beginners and 21 experienced people, nine "
        "combinations of stance width and foot angle, no heavy loads) recommends a moderate foot "
        "angle of about 20 degrees with feet about shoulder-width apart. The extreme combinations "
        "(narrow stance with feet turned out 42 degrees; wide stance with feet straight) gave the "
        "largest loads. Knees falling inwards were rare in healthy participants (seen with a wide "
        "stance); beginners' knees tended to be further out than those of experienced people.",
        "- Both sources studied healthy people or discuss rehabilitation; they do not prove that a "
        "given tip prevents injuries.",
    ]))
    bg.append(("COACH'S TIPS IN THIS REPORT", [
        "- Keeping the heels flat and going to thigh-level are common coaching advice. The studies "
        "I read do not test them, which is why they are labelled 'coach's tip'.",
    ]))
    sources = source_lines(items)
    bg.append(("SOURCES", [
        "- Straub and Powers (2024), International Journal of Sports Physical Therapy: 'A "
        "Biomechanical Review of the Squat Exercise: Implications for Clinical Practice' (full "
        "text read).",
        "- Lorenzetti et al. (2018), BMC Sports Science, Medicine and Rehabilitation: 'How to "
        "squat? Effects of various stance widths, foot placement angles and level of experience "
        "on knee, hip and trunk motion and loading' (full text read).",
    ] + sources))
    return main, bg


# ----------------------------------- the analysis -------------------------------

def stat_line(label, values, unit, decimals=0):
    """Print one measurement as: median (middle half of the values, count)."""
    med, lo, hi, n = summarize(values)
    print(f"  {label}: {med:.{decimals}f} {unit} (middle half: {lo:.{decimals}f} to "
          f"{hi:.{decimals}f}, {n} values)")


def analyze_video(video, speed_kmh=None, output_dir="data/output", recording="right"):
    """Run the squat analysis on one video (one side of the lifter facing the camera).

    `recording` is "left" or "right": the side of the body that faces the camera.
    `speed_kmh` is not used for the squat, it is kept so that every sport has the same signature.

    Returns a dictionary like running.analyze_video, plus "background" (the technical part of
    the report). Raises FileNotFoundError or ValueError (with a readable message) if the video
    cannot be analyzed.
    """
    if recording not in SIDES:
        raise ValueError("Choose which side of your body faces the camera: left or right.")
    prefix = SIDES[recording]

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    details = io.StringIO()
    with contextlib.redirect_stdout(details):
        fps, points = extract_landmarks(video)
        if len(points) == 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not read any frame from the video.")
        duration = len(points) / fps
        if duration < MIN_DURATION_S:
            raise ValueError(f"The video is too short: at least {MIN_DURATION_S:.0f} seconds "
                             f"with {MIN_REPS} or more squats are needed.")
        leg = np.stack([landmark(points, prefix, n) for n in ("hip", "knee", "ankle")], axis=1)
        detected = float(np.mean(~np.isnan(leg).any(axis=(1, 2))))
        if detected < MIN_DETECTED:
            raise ValueError(f"Your {recording} leg was found in only {detected:.0%} of the "
                             "frames. Make sure your whole body is visible from the side and the "
                             "light is good.")

        direction = facing_direction(points, prefix)
        bottoms, hip_y = find_bottoms(points, fps, prefix)
        starts = find_starts(hip_y, bottoms)
        measures = analyze_reps(points, fps, prefix, direction, bottoms, starts)
        reps = len(measures["flex"])

        print(f"Video: {duration:.1f}s at {fps:.0f} fps")
        print(f"Analysed side: {recording}")
        print(f"Lifter facing the {'right' if direction == 1 else 'left'} of the image")
        print(f"Leg detected in {detected:.0%} of the frames")
        print(f"Lowest points found: {len(bottoms)}; squats used: {reps}")
        stat_line("Knee bend at the bottom", measures["flex"], "deg")
        stat_line("Chest lean from upright", measures["trunk"], "deg")
        stat_line("Shin lean from upright", measures["tibia"], "deg")
        stat_line("Chest lean minus shin lean", measures["diff"], "deg")
        stat_line("Heel rise", [h * 100 for h in measures["heel"]], "% of foot length")

    if reps < MIN_REPS:
        raise ValueError(f"I found only {reps} squat(s); at least {MIN_REPS} are needed. Film a "
                         "whole set from the side, with your whole body in the picture, and "
                         "squat down by at least a half.")

    # Video with only the analysed side; a ring on the hip marks the lowest point of each squat
    out_dir = Path(output_dir)
    raw_path = out_dir / f"{video.stem}_skeleton_raw.mp4"
    final_path = out_dir / f"{video.stem}_skeleton.mp4"
    none = np.array([], dtype=int)
    ring_frames = max(3, int(round(0.3 * fps)))
    events = (bottoms, none) if prefix == "l" else (none, bottoms)
    render_skeleton(video, raw_path, fps, points, events[0], events[1],
                    flash_frames=ring_frames, only_side=prefix,
                    ring_landmark="hip", ring_label="")
    if to_h264(raw_path, final_path):
        raw_path.unlink()
    else:
        raw_path.replace(final_path)

    items = build_advice(measures)
    main_sections, background = build_report(fps, duration, reps, measures, items)
    return {
        "sections": main_sections,
        "background": background,
        "report_text": report_to_text(main_sections + background),
        "details": details.getvalue(),
        "skeleton_path": final_path,
        "fps": fps,
        "duration": duration,
    }


def main():
    """Command-line use: python -m src.sports.squat VIDEO [left|right]"""
    args = sys.argv[1:]
    video = Path(args[0]) if args else Path("data/input/squat_side.mp4")
    side = args[1] if len(args) > 1 else "right"

    print("Analyzing the video (this can take a minute)...")
    try:
        result = analyze_video(video, recording=side)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    print(result["details"])
    print(f"Skeleton video saved to: {result['skeleton_path']}\n")
    print(result["report_text"])


if __name__ == "__main__":
    main()