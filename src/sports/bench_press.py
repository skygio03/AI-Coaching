"""Flat bench press analysis from the FOOT END of the bench, with the camera raised.

The camera looks at the lifter from beyond the feet, along the bench, above the lifter's
head height, tilted down to see the chest, both arms and the bar. This view shows how wide
the hands are compared with the shoulders and whether the bar stays level. It cannot show
the shoulder blades or the path of the bar, so those are given as "check it yourself" tips.
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
MIN_REPS = 3                # assumption: fewer repetitions give too little data
MIN_DURATION_S = 5.0        # assumption
MIN_DETECTED = 0.7          # assumption: share of frames in which shoulders and wrists must be found
MIN_TRAVEL = 0.25           # assumption: wrists must move by at least 0.25 shoulder widths
MIN_REP_TIME_S = 1.0        # assumption: two repetitions cannot be closer than this
WINDOW_S = 0.07             # assumption: values are read around the key frame (median)

# Hand width compared with shoulder width. The studies tested 1, 1.5 and 2 times the width
# between the shoulder bones (Noteboom et al. 2024); the limits below are my own choice,
# placed between those values to leave room for the measurement error of a 2D video.
WIDE_GRIP_RATIO = 1.75      # assumption
NARROW_GRIP_RATIO = 1.2     # assumption
GRIP_RATIO_VALID = (0.8, 3.5)   # assumption: outside this range the measurement is not trusted
MAX_RATIO_SPREAD = 0.3      # assumption: if the middle half of the repetitions is wider, no advice

BAR_TILT_DEG = 5            # assumption: a tilt below this is within the measurement error
TILT_CONSISTENCY = 0.7      # assumption: share of repetitions that must tilt the same way

# Interface: one camera position, shown so that the user knows how to film
RECORDING_OPTIONS = {
    "Camera at the foot end of the bench, raised (the only view supported)": "foot_end",
}
OPTION_QUESTION = "Where is the camera?"
VIDEO_NOTE = "Both sides are drawn. A ring on each wrist marks the lowest point of each repetition."
NOTICE = ("The bench press analysis is the least tested one: the pose model finds lying people "
          "less reliably, and this camera view has not been tried on real videos yet. Check "
          "that the skeleton follows your arms and shoulders, and treat the numbers with caution.")

REQUIRED = ("l_shoulder", "r_shoulder", "l_wrist", "r_wrist")


# ------------------------------- helpers -------------------------------------

def nanmed(values):
    """Median ignoring NaN; NaN if there is nothing to take the median of."""
    a = np.asarray(values, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmedian(a))


def landmark(points, name):
    """Pixel positions of one landmark, shape [n_frames, 2]."""
    return points[:, NAMES.index(name)]


def line_angle(left, right):
    """Angle (degrees) of the line from `left` to `right` in the image, between -90 and 90.

    The line is turned so that it always points to the right of the image, so the result does
    not change if the pose model swaps left and right.
    """
    d = right - left
    flip = np.where(d[:, 0] < 0, -1.0, 1.0)
    return np.degrees(np.arctan2(d[:, 1] * flip, d[:, 0] * flip))


def shoulder_width_px(points):
    """Median distance between the two shoulders, in pixels."""
    return nanmed(np.linalg.norm(landmark(points, "l_shoulder") - landmark(points, "r_shoulder"),
                                 axis=1))


def bar_signal(points, fps):
    """Wrist height minus shoulder height, in shoulder widths (image y grows downwards).

    Large values: the wrists are low in the picture, close to the chest (bottom of the lift).
    Small values: the arms are extended (lockout).
    """
    wrists = (landmark(points, "l_wrist")[:, 1] + landmark(points, "r_wrist")[:, 1]) / 2
    shoulders = (landmark(points, "l_shoulder")[:, 1] + landmark(points, "r_shoulder")[:, 1]) / 2
    d = (wrists - shoulders) / shoulder_width_px(points)
    return smooth(fill_gaps(d), fps)


def find_bottoms(signal, fps):
    """Frames where the bar is closest to the chest."""
    peaks, _ = find_peaks(signal, prominence=MIN_TRAVEL,
                          distance=max(1, int(MIN_REP_TIME_S * fps)))
    return peaks


def find_tops(signal, bottoms):
    """For each repetition, the frame where the arms were most extended before it."""
    tops, previous = [], 0
    for b in bottoms:
        tops.append(previous + int(np.argmin(signal[previous:b + 1])))
        previous = b
    return tops


def analyze_reps(points, fps, bottoms, tops):
    """Measure every repetition. Returns a dict of lists (one value per repetition)."""
    lw, rw = landmark(points, "l_wrist"), landmark(points, "r_wrist")
    ls, rs = landmark(points, "l_shoulder"), landmark(points, "r_shoulder")
    hand_gap = np.abs(rw[:, 0] - lw[:, 0])
    shoulder_gap = np.abs(rs[:, 0] - ls[:, 0])
    tilt = line_angle(lw, rw) - line_angle(ls, rs)
    w = max(1, int(round(WINDOW_S * fps)))

    out = {"grip": [], "tilt": [], "seconds": []}
    previous = None
    for t, b in zip(tops, bottoms):
        low = slice(max(b - w, 0), b + w + 1)
        top = slice(max(t - w, 0), t + w + 1)
        out["grip"].append(nanmed(hand_gap[low]) / nanmed(shoulder_gap[low]))
        out["tilt"].append(nanmed(tilt[top]))
        if previous is not None:
            out["seconds"].append((b - previous) / fps)
        previous = b
    return out


# ------------------------------- advice --------------------------------------

def stable(values, limit=None):
    """Return (median, ok): ok if there are enough repetitions and they are similar."""
    med, lo, hi, n = summarize(values)
    ok = n >= MIN_REPS and (limit is None or (hi - lo) <= limit)
    return med, ok


def check_items():
    """Things this view cannot show: the person has to check them alone."""
    return [
        Advice(
            CHECK, "Shoulder blades back and down",
            try_this="Before you take the bar off the rack, pull your shoulder blades together "
                     "and down into the bench, and keep them there for the whole set.",
            why="In a computer-model study of 10 experienced lifters, pulling the shoulder blades "
                "back lowered the load on the shoulder joint and the work of the small muscles "
                "that hold it in place. The chest muscles were less active in the model. The authors "
                "cite another study in which it did not change how much weight people lifted.",
            source="Noteboom et al. 2024.",
        ),
        Advice(
            CHECK, "Try pushing your hands outwards, as if bending the bar",
            try_this="While you press, push your hands apart on the bar, as if you wanted to bend "
                     "it around your chest. Treat this as an idea to try, not as a rule.",
            why="A computer simulation found that pushing outwards on the bar lowered the load on "
                "the shoulder joints. It was a simulation of one lift, not a test of this tip "
                "with real lifters. The two least experienced lifters in that study pushed "
                "inwards instead.",
            source="Noteboom et al. 2024 (the authors say the numbers cannot be generalised).",
        ),
    ]


def build_advice(measures):
    """Turn the measures into tips. Only tips supported by enough similar repetitions are given."""
    items = []

    ratio, ratio_ok = stable(measures["grip"], MAX_RATIO_SPREAD)
    valid = np.isfinite(ratio) and GRIP_RATIO_VALID[0] <= ratio <= GRIP_RATIO_VALID[1]
    if ratio_ok and valid and ratio >= WIDE_GRIP_RATIO:
        items.append(Advice(
            RESEARCH, "Your hands look wide on the bar",
            saw=f"Your hands are about {ratio:.1f} times as far apart as your shoulders.",
            try_this="If you have shoulder pain, or you do not need to lift the most weight "
                     "possible, try moving your hands in a little, towards about 1.5 times the "
                     "width of your shoulders. A wider grip lets you lift more, so this is a "
                     "choice between a stronger lift and an easier time for your shoulders.",
            why="In a study of 10 experienced lifters, a very wide grip (twice the width of the "
                "shoulders) put clearly more load on the collarbone joint and on the muscles "
                "that hold the shoulder in place than a grip of 1.5 times or less. Another "
                "study found that wide and medium grips lift more weight than a narrow one.",
            sure="low", priority=2,
            source="Noteboom et al. 2024 (computer model, 16 kg bar); Larsen et al. 2021 "
                   "(maximum lifts, recreationally trained men).",
        ))
    elif ratio_ok and valid and ratio <= NARROW_GRIP_RATIO:
        items.append(Advice(
            RESEARCH, "Your hands look narrow on the bar",
            saw=f"Your hands are about {ratio:.1f} times as far apart as your shoulders.",
            try_this="If you want to lift the most weight, a grip a little wider, towards about "
                     "1.5 times the width of your shoulders, lets you lift more. If you feel pain "
                     "at the top of your shoulder, avoid a very narrow grip.",
            why="In one study people lifted less with a narrow grip than with a medium or wide "
                "one. In another, a very narrow grip (as wide as the shoulders) pushed the "
                "upper arm bone up more against the shoulder, which may matter if the top of the "
                "shoulder hurts.",
            sure="low", priority=2,
            source="Larsen et al. 2021; Noteboom et al. 2024.",
        ))

    tilt, tilt_ok = stable(measures["tilt"])
    if tilt_ok and np.isfinite(tilt) and abs(tilt) >= BAR_TILT_DEG:
        values = np.array([t for t in measures["tilt"] if np.isfinite(t)])
        same = float(np.mean(np.sign(values) == np.sign(tilt)))
        if same >= TILT_CONSISTENCY:
            items.append(Advice(
                COACH, "One hand ends higher than the other",
                saw=f"At the top of the lift the line between your hands is tilted by about "
                    f"{abs(tilt):.0f} degrees compared with your shoulders, in "
                    f"{same:.0%} of the repetitions.",
                try_this="Push both hands up at the same speed so the bar stays level. If one "
                         "side is weaker, use a lighter weight until both sides move evenly, "
                         "and film yourself again.",
                why="A tilted bar usually means one side is doing more of the work.",
                sure="low", priority=3,
                source="Common coaching advice. The studies I read did not test it.",
            ))

    return items + check_items()


# ------------------------------- the report ----------------------------------

def filming_tips(recording="foot_end"):
    """Return the filming tips for the bench press, as a list of '- ' lines."""
    return [
        "- Put the camera at the foot end of the bench, in line with the bench and with you, "
        "3 to 4 metres away. Your head is towards the rack.",
        "- Raise the camera to about the height of your chest when standing (1.3 to 1.6 metres) "
        "and tilt it down, so that it sees your chest, both arms and both hands.",
        "- Your shoulders, elbows, wrists and the bar must stay in the picture during the whole "
        "lift. Your knees and the rack should not hide them.",
        "- Tight-fitting top, good light, nothing between the camera and your arms.",
        f"- Film a whole set: at least {MIN_REPS} repetitions (5 or more is better), without "
        "moving the camera.",
        "- Use a spotter or the safety arms of the rack for heavy sets, and keep the camera out "
        "of their way.",
        "- 60 frames per second or more if your phone allows it.",
        "- This camera position is a suggestion of this project and has not been tested yet.",
    ]


def build_report(fps, duration, reps, measures, items):
    """Return (main_sections, background_sections) as lists of (title, [lines])."""
    ratio, ratio_ok = stable(measures["grip"], MAX_RATIO_SPREAD)
    valid = np.isfinite(ratio) and GRIP_RATIO_VALID[0] <= ratio <= GRIP_RATIO_VALID[1]
    tilt, _, _, _ = summarize(measures["tilt"])
    seconds, _, _, _ = summarize(measures["seconds"])

    main = [("YOUR RESULT", [
        f"Video: {duration:.1f} s at {fps:.0f} fps, {reps} repetitions counted.",
        "These are estimates from a single 2D video: not a medical opinion and not a coach.",
    ])]
    main.append(what_to_do_section(
        items, "Nothing stands out in what I can measure from this view. Still go through the "
               "list below, because the camera cannot see the most important points."))

    measured = [f"- Repetitions counted: {reps}."]
    if valid:
        measured.append(f"- Hands: about {ratio:.1f} times as far apart as your shoulders "
                        "(the studies compared 1, 1.5 and 2 times).")
    else:
        measured.append("- Hands: I could not measure the width of your grip reliably.")
    if np.isfinite(tilt):
        measured.append(f"- Bar: at the top the line between your hands is tilted by about "
                        f"{abs(tilt):.0f} degrees compared with your shoulders.")
    if np.isfinite(seconds):
        measured.append(f"- Pace: about {seconds:.1f} seconds per repetition.")
    main.append(("WHAT I MEASURED", measured))
    main.append(check_section(items))

    trust = [f"- {reps} repetitions were analysed (at least {MIN_REPS} are needed).",
             "- The pose model finds lying people less reliably than standing ones: check that "
             "the skeleton follows your shoulders and wrists.",
             "- The grip width is the distance between your wrists compared with the distance "
             "between your shoulder joints, seen from a raised camera: it is only approximate, "
             "which is why I give advice only if your hands are clearly wide or narrow."]
    if reps >= MIN_REPS and not ratio_ok:
        trust.append("- The width of your hands changes a lot between repetitions in the video, "
                     "so I did not give advice about it.")
    main.append(("HOW MUCH TO TRUST THIS", trust))

    bg = []
    extra = extra_section(items)
    if extra:
        bg.append(extra)
    bg.append(("THE NUMBERS", [
        f"- Hand distance divided by shoulder distance: {fmt(measures['grip'], 2)}",
        f"- Tilt of the line between the hands compared with the shoulders at the top: "
        f"{fmt(measures['tilt'], 1)} degrees",
        f"- Time per repetition: {fmt(measures['seconds'], 1)} s",
        "  'from X to Y' is the middle half of the repetitions, i.e. how much the value changes "
        "from one repetition to the next.",
    ]))
    bg.append(("HOW THE TIPS WERE DECIDED", [
        f"- Hands wide: ratio of {WIDE_GRIP_RATIO} or more; narrow: {NARROW_GRIP_RATIO} or less. "
        "The studies used 1, 1.5 and 2 times the width between the shoulder bones; the limits "
        "are my own choice, between those values, to leave room for the measurement error.",
        f"- Bar tilt: given only if the tilt is at least {BAR_TILT_DEG} degrees in at least "
        f"{TILT_CONSISTENCY:.0%} of the repetitions (assumptions).",
        f"- A tip is given only if at least {MIN_REPS} repetitions were found and, for the hands, "
        f"the middle half of the repetitions is within {MAX_RATIO_SPREAD} of the ratio "
        "(assumption).",
        "- Repetitions are found from the movement of the wrists compared with the shoulders: "
        f"a repetition counts if they move by at least {MIN_TRAVEL} shoulder widths "
        "(assumption).",
        "- The pose model gives the position of the shoulder joints, not of the bone tips used "
        "in the studies, and the wrists, not the exact point where the hand holds the bar.",
    ]))
    bg.append(("WHAT CANNOT BE MEASURED FROM THIS VIDEO", [
        "- Whether the shoulder blades are pulled back (shown as a 'check it yourself' tip).",
        "- The path of the bar and where it touches the chest.",
        "- How far the elbows point out from the body (the view along the bench squashes it).",
        "- The push of the hands inwards or outwards on the bar.",
        "- The weight on the bar and the speed of the bar.",
    ]))
    bg.append(("WHAT THE RESEARCH SAYS", [
        "- A 2024 study (Noteboom et al.; 10 experienced lifters, 9 men and 1 woman, a light 16 kg "
        "bar, loads estimated by a computer model of the shoulder, not injuries) found that a "
        "grip narrower than 1.5 times the width between the shoulder bones lowered the pressure "
        "on the collarbone joint and the backward push on the shoulder joint, and that pulling "
        "the shoulder blades back lowered the load on the shoulder joint and the work of the "
        "rotator cuff. The angle of the upper arm from the body had a smaller effect; an upper "
        "arm at 45 degrees from the body, or a very narrow grip (as wide as the shoulders), "
        "gave a larger upward push on the shoulder joint. 'Releasing' the shoulder blades did "
        "not differ from neutral.",
        "- In the same study the sideways force of the hands on the bar varied a lot between "
        "lifters. In a simulation of one lift, pushing outwards lowered the loads; the authors "
        "say this cannot be generalised. They suggest feedback on the direction of the force as "
        "something to study.",
        "- A 2021 study (Larsen et al.; recreationally trained men, maximal lift) found that wide "
        "and medium grips let people lift more than a narrow grip. So there is a trade-off "
        "between lifting more and the load on the shoulder.",
        "- A 2022 study (Schütz et al.; 20 healthy people, moderate loads) found lower and "
        "steadier loads at the shoulder with the bench press than with cable cross-overs. It "
        "does not compare techniques of the bench press, so it is not used for the tips.",
        "- None of these studies followed people over time, so none shows that a technique "
        "causes or prevents an injury.",
    ]))
    bg.append(("COACH'S TIPS IN THIS REPORT", [
        "- Keeping the bar level is common coaching advice. The studies I read do not test it, "
        "which is why it is labelled 'coach's tip'.",
    ]))
    sources = source_lines(items)
    bg.append(("SOURCES", [
        "- Noteboom et al. (2024), Frontiers in Physiology: 'Effects of bench press technique "
        "variations on musculoskeletal shoulder loads and potential injury risk' (full text "
        "read).",
        "- Larsen, Gomo and van den Tillaar (2021), Frontiers in Sports and Active Living: 'A "
        "Biomechanical Analysis of Wide, Medium, and Narrow Grip Width Effects on Kinematics, "
        "Horizontal Kinetics, and Muscle Activity on the Sticking Region in Recreationally "
        "Trained Males During 1-RM Bench Pressing' (full text read).",
        "- Schütz et al. (2022), Sports: 'Chest Exercises: Movement and Loading of Shoulder, "
        "Elbow and Wrist Joints' (full text read).",
    ] + sources))
    return main, bg


# ----------------------------------- the analysis -------------------------------

def stat_line(label, values, unit, decimals=0):
    """Print one measurement as: median (middle half of the values, count)."""
    med, lo, hi, n = summarize(values)
    print(f"  {label}: {med:.{decimals}f} {unit} (middle half: {lo:.{decimals}f} to "
          f"{hi:.{decimals}f}, {n} values)")


def analyze_video(video, speed_kmh=None, output_dir="data/output", recording="foot_end"):
    """Run the bench press analysis on one video filmed from the foot end of the bench.

    `recording` must be "foot_end": it is kept so that every sport has the same signature.
    `speed_kmh` is not used for the bench press, for the same reason.

    Returns a dictionary like running.analyze_video, plus "background" (the technical part of
    the report). Raises FileNotFoundError or ValueError (with a readable message) if the video
    cannot be analyzed.
    """
    if recording != "foot_end":
        raise ValueError("Only the camera at the foot end of the bench is supported for now.")

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
                             f"with {MIN_REPS} or more repetitions are needed.")
        needed = np.stack([landmark(points, n) for n in REQUIRED], axis=1)
        detected = float(np.mean(~np.isnan(needed).any(axis=(1, 2))))
        if detected < MIN_DETECTED:
            raise ValueError(f"Both shoulders and both wrists were found in only {detected:.0%} "
                             "of the frames. Check the camera position (foot end of the bench, "
                             "raised), the light, and that nothing hides your arms.")

        signal = bar_signal(points, fps)
        bottoms = find_bottoms(signal, fps)
        tops = find_tops(signal, bottoms)
        measures = analyze_reps(points, fps, bottoms, tops)
        reps = len(bottoms)

        print(f"Video: {duration:.1f}s at {fps:.0f} fps")
        print(f"Shoulders and wrists detected in {detected:.0%} of the frames")
        print(f"Repetitions found: {reps}")
        stat_line("Hand distance / shoulder distance", measures["grip"], "", 2)
        stat_line("Tilt of the hands compared with the shoulders", measures["tilt"], "deg", 1)
        stat_line("Time per repetition", measures["seconds"], "s", 1)

    if reps < MIN_REPS:
        raise ValueError(f"I found only {reps} repetition(s); at least {MIN_REPS} are needed. "
                         "Film a whole set from the foot end of the bench, with the camera "
                         "raised, and your shoulders and wrists always in the picture.")

    # Video with both sides drawn; a ring on each wrist marks the lowest point of each repetition
    out_dir = Path(output_dir)
    raw_path = out_dir / f"{video.stem}_skeleton_raw.mp4"
    final_path = out_dir / f"{video.stem}_skeleton.mp4"
    ring_frames = max(3, int(round(0.3 * fps)))
    render_skeleton(video, raw_path, fps, points, bottoms, bottoms,
                    flash_frames=ring_frames, ring_landmark="wrist", ring_label="")
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
    """Command-line use: python -m src.sports.bench_press VIDEO"""
    args = sys.argv[1:]
    video = Path(args[0]) if args else Path("data/input/bench_foot_end.mp4")

    print("Analyzing the video (this can take a minute)...")
    try:
        result = analyze_video(video)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    print(result["details"])
    print(f"Skeleton video saved to: {result['skeleton_path']}\n")
    print(result["report_text"])


if __name__ == "__main__":
    main()