"""Running analysis (side view): cadence, stride mechanics and posture."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.common.advice import (CHECK, COACH, Advice, check_section, extra_section,
                               source_lines, what_to_do_section)
from src.common.pose_extractor import extract_landmarks, NAMES
from src.common.render import render_skeleton, to_h264

# ---------------------------------------------------------------------------
# Settings. Values marked "assumption" are my own choices, NOT taken from the
# paper (which gives no thresholds); check them against the video.
# ---------------------------------------------------------------------------
TOUCHDOWN_LEVEL = 0.83   # how close to the ground the foot must be to count as touchdown
LOADING_DELAY_S = 0.05   # assumption: loading response happens ~50 ms after touchdown

# Report rules (all assumptions)
MIN_FPS_FOR_EVENT_MEASURES = 100  # below this, landing and tibia angle depend on the exact frame
MIN_STRIKES_PER_LEG = 10          # fewer than 10 gait cycles is unwise for small effects (Fox 2023)
MIN_DURATION_S = 10.0             # shorter clips give too few steps

# How the video was filmed: label shown in the interface -> internal key
# Step rate reference by speed: 28 regular runners on a treadmill, 3D motion capture
# (Fukuchi et al. 2017, Table 4). Strides per minute x 2 = steps per minute, SD x 2.
# (speed km/h, steps/min, SD). It describes a group, it is NOT a target or a cut-off.
CADENCE_REFERENCE = ((9.0, 161.6, 9.3), (12.6, 171.4, 10.5), (16.2, 183.5, 13.4))
LOW_CADENCE_SD = 1.0              # assumption: "on the slow side" = more than 1 SD below the group mean
STEP_RATE_CHANGE = 0.05           # 5%: the small end of the 5-10% used in the studies

RECORDING_OPTIONS = {
    "Treadmill (camera still)": "treadmill",
    "Outdoors, camera following the runner": "following",
}
SPEED_HELP = {
    "treadmill": "The speed shown on the treadmill display. Leave at 0 if you do not know it.",
    "following": "From your watch/GPS, or distance divided by time. Leave at 0 if you do not "
                 "know it.",
}


# ------------------------------- basic helpers -------------------------------

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


def running_direction(points):
    """Return +1 if the runner faces the right of the image, -1 if the left.

    The toe is always ahead of the heel in the direction of travel.
    """
    diffs = []
    for foot in ("l", "r"):
        heel_x = points[:, NAMES.index(f"{foot}_heel"), 0]
        toe_x = points[:, NAMES.index(f"{foot}_toe"), 0]
        diffs.append(toe_x - heel_x)
    return 1 if np.nanmedian(np.concatenate(diffs)) >= 0 else -1


def leg_length(points, foot):
    """Median thigh + shank length in pixels over the whole video."""
    hip = points[:, NAMES.index(f"{foot}_hip")]
    knee = points[:, NAMES.index(f"{foot}_knee")]
    ankle = points[:, NAMES.index(f"{foot}_ankle")]
    lengths = np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)
    return float(np.nanmedian(lengths))


# ------------------------------ foot strike events ----------------------------

def foot_height_signal(points, fps, foot):
    """Vertical position of the lowest point of the foot ('l' or 'r').

    In image coordinates y grows downward, so a larger value means a lower foot.
    """
    heel = points[:, NAMES.index(f"{foot}_heel"), 1]
    toe = points[:, NAMES.index(f"{foot}_toe"), 1]
    # The lowest point of the foot is the max y; works for heel and forefoot strikers
    lowest = np.fmax(heel, toe)
    return smooth(fill_gaps(lowest), fps)


def detect_strikes(points, fps, foot, level=TOUCHDOWN_LEVEL, window_s=0.3, ground_window_s=1.0):
    """Return (touchdowns, toe_offs) as arrays of frame indices for one foot.

    The foot height has a plateau while the foot is on the ground. The plateau height
    ("ground") is estimated locally around each strike, so it follows the runner if
    they drift in the frame (or the camera moves). Starting from the middle of each
    plateau, walk outwards while the foot stays near the ground:
    - touchdown = first frame of the run at `level` of the way between the foot's
      highest point in the window and the ground height
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

        # Local ground height: 80th percentile of the foot height within +/- 1 s
        lo = max(0, p - int(ground_window_s * fps))
        hi = min(len(y), p + int(ground_window_s * fps) + 1)
        ground = np.percentile(y[lo:hi], 80)

        # The ground can never be above the peak itself, or the window would collapse
        ground = min(ground, y[p])
        threshold = top + level * (ground - top)

        # Stop at the first frame that leaves the plateau
        a = p
        while a > start and y[a - 1] >= threshold:
            a -= 1
        b = p
        while b < end and y[b + 1] >= threshold:
            b += 1

        touchdowns.append(a)
        toe_offs.append(b)

    return np.array(touchdowns, dtype=int), np.array(toe_offs, dtype=int)


def loading_delay_frames(fps):
    """Number of frames between touchdown and the loading response (at least 1)."""
    return max(1, int(round(LOADING_DELAY_S * fps)))


def loading_frame(touchdown, toe_off, fps):
    """Frame of the loading response: shortly after touchdown, never after toe-off."""
    return min(touchdown + loading_delay_frames(fps), toe_off)


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


# ------------------------------- measurements ---------------------------------

def knee_angle_at(points, f, foot):
    """Hip-knee-ankle angle at frame f (180 = straight leg); NaN if a landmark is missing."""
    hip, knee, ankle = (points[f, NAMES.index(f"{foot}_{n}")]
                        for n in ("hip", "knee", "ankle"))
    if np.isnan([hip, knee, ankle]).any():
        return float("nan")
    return joint_angle(hip, knee, ankle)


def landing_offset(points, f, foot, direction, length):
    """Horizontal distance from the pelvis centre to the ankle at frame f.

    Expressed as a fraction of leg length; positive = ankle in front of the pelvis,
    in the running direction. Returns NaN if a landmark is missing.
    """
    l_hip_x = points[f, NAMES.index("l_hip"), 0]
    r_hip_x = points[f, NAMES.index("r_hip"), 0]
    ankle_x = points[f, NAMES.index(f"{foot}_ankle"), 0]
    if np.isnan([l_hip_x, r_hip_x, ankle_x]).any():
        return float("nan")
    pelvis_x = (l_hip_x + r_hip_x) / 2
    return direction * (ankle_x - pelvis_x) / length


def tibia_angle(points, f, foot, direction):
    """Signed angle of the shank from vertical, in degrees.

    Positive = knee in front of the ankle ("flexed tibia"); negative = knee behind
    the ankle ("extended tibia"); about 0 = vertical tibia.
    """
    knee = points[f, NAMES.index(f"{foot}_knee")]
    ankle = points[f, NAMES.index(f"{foot}_ankle")]
    if np.isnan([knee, ankle]).any():
        return float("nan")
    dx = direction * (knee[0] - ankle[0])
    dy = ankle[1] - knee[1]                 # the knee is above the ankle, so dy > 0
    return float(np.degrees(np.arctan2(dx, dy)))


def trunk_lean_series(points, direction):
    """Trunk angle from vertical for every frame; positive = leaning forward.

    Uses the line from the hip midpoint to the shoulder midpoint.
    """
    shoulders = (points[:, NAMES.index("l_shoulder")] + points[:, NAMES.index("r_shoulder")]) / 2
    hips = (points[:, NAMES.index("l_hip")] + points[:, NAMES.index("r_hip")]) / 2
    dx = direction * (shoulders[:, 0] - hips[:, 0])
    dy = hips[:, 1] - shoulders[:, 1]       # shoulders are above the hips, so dy > 0
    return np.degrees(np.arctan2(dx, dy))


def vertical_oscillation(points, fps, all_touchdowns, length):
    """Vertical range of the pelvis within each step, as % of leg length.

    Measured inside each step (between two consecutive touchdowns), so a slow drift
    of the camera or of the runner in the frame does not matter.
    """
    pelvis_y = (points[:, NAMES.index("l_hip"), 1] + points[:, NAMES.index("r_hip"), 1]) / 2
    pelvis_y = smooth(fill_gaps(pelvis_y), fps)

    values = []
    for a, b in zip(all_touchdowns[:-1], all_touchdowns[1:]):
        # Skip intervals that are not one step (a missed or doubled strike)
        if not 0.2 <= (b - a) / fps <= 0.7:
            continue
        segment = pelvis_y[a:b + 1]
        values.append((segment.max() - segment.min()) / length * 100)
    return values


def sensitivity(points, td_ok, off_ok, foot, direction, length):
    """Median landing offset and tibia angle if the loading frame is 1, 2 or 3 frames
    after touchdown. A large spread means the measure depends on timing, not on form."""
    rows = []
    for k in (1, 2, 3):
        frames = [min(a + k, b) for a, b in zip(td_ok, off_ok)]
        land = [landing_offset(points, f, foot, direction, length) * 100 for f in frames]
        tib = [tibia_angle(points, f, foot, direction) for f in frames]
        rows.append((k, float(np.nanmedian(land)), float(np.nanmedian(tib))))
    return rows


def analyze_leg(points, fps, foot, td, off, direction):
    """Measure one leg and return the results as a dict (nothing is printed here)."""
    # Ground contact time in ms; values outside 100-500 ms are detection errors
    contact = np.array([(b - a) / fps * 1000 for a, b in zip(td, off)])
    plausible = (contact >= 100) & (contact <= 500)
    discarded = [f"{a / fps:.1f}s ({(b - a) / fps * 1000:.0f} ms)"
                 for a, b, ok in zip(td, off, plausible) if not ok]

    td_ok, off_ok = td[plausible], off[plausible]
    length = leg_length(points, foot)

    # Knee flexion (0 deg = straight leg), the convention used in the paper
    flex_touchdown = [180 - knee_angle_at(points, f, foot) for f in td_ok]
    peak_flexion = []
    for a, b in zip(td_ok, off_ok):
        angles = [knee_angle_at(points, f, foot) for f in range(a, b + 1)]
        angles = [x for x in angles if not np.isnan(x)]
        peak_flexion.append(180 - min(angles) if angles else float("nan"))
    excursion = list(np.array(peak_flexion) - np.array(flex_touchdown))

    # Overstride and tibia angle, both read at the loading response
    lr = [loading_frame(a, b, fps) for a, b in zip(td_ok, off_ok)]
    landing = [landing_offset(points, f, foot, direction, length) * 100 for f in lr]
    tibia = [tibia_angle(points, f, foot, direction) for f in lr]

    return {
        "strikes_total": len(td),
        "strikes_used": len(td_ok),
        "discarded": discarded,
        "flex_touchdown": flex_touchdown,
        "peak_flexion": peak_flexion,
        "excursion": excursion,
        "landing": landing,
        "tibia": tibia,
        "sensitivity": (sensitivity(points, td_ok, off_ok, foot, direction, length)
                        if len(td_ok) else []),
    }


# ------------------------------ technical output -------------------------------

def summarize(values):
    """Return (median, 25th percentile, 75th percentile, count), ignoring NaN."""
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), len(v))


def report(label, values, unit, decimals=0, what="strikes"):
    """Print one measurement as: median (middle half of the values, count)."""
    med, lo, hi, n = summarize(values)
    d = decimals
    print(f"  {label}: {med:.{d}f} {unit} (middle half: {lo:.{d}f} to {hi:.{d}f}, {n} {what})")


def print_details(label, leg):
    """Print the technical numbers of one leg."""
    print(f"\n{label} leg")
    if leg["discarded"]:
        print(f"  {len(leg['discarded'])} of {leg['strikes_total']} strikes discarded "
              f"(implausible stance window) at: {', '.join(leg['discarded'])}")
    report("Knee flexion at touchdown", leg["flex_touchdown"], "deg")
    report("Knee peak flexion in stance", leg["peak_flexion"], "deg")
    report("Knee excursion (touchdown to peak)", leg["excursion"], "deg")
    report("Ankle vs pelvis at loading response", leg["landing"], "% of leg length")
    report("Tibia angle at loading response (knee in front = positive)", leg["tibia"], "deg")
    if leg["sensitivity"]:
        print("  Sensitivity to the loading frame (landing vs pelvis, tibia angle):")
        for k, land, tib in leg["sensitivity"]:
            print(f"    +{k} frame: {land:.0f} % of leg length, {tib:.0f} deg")


# ---------------------------------- the report ----------------------------------

def reference_cadence(speed_kmh):
    """Return (mean, sd) in steps/min of the lab reference group at this speed, or None.

    Linear interpolation between the three speeds of the study; None outside 8-17 km/h or
    when the speed is unknown.
    """
    if not speed_kmh or not 8.0 <= speed_kmh <= 17.0:
        return None
    speeds = [r[0] for r in CADENCE_REFERENCE]
    mean = float(np.interp(speed_kmh, speeds, [r[1] for r in CADENCE_REFERENCE]))
    sd = float(np.interp(speed_kmh, speeds, [r[2] for r in CADENCE_REFERENCE]))
    return mean, sd


def landing_summary(leg):
    """Return (lowest, highest, all_ahead) of the median landing offset over the 1-, 2-
    and 3-frame readings, or None if there is no data.

    all_ahead = True if the ankle is in front of the pelvis centre in every reading.
    """
    values = [land for _, land, _ in leg["sensitivity"] if not np.isnan(land)]
    if not values:
        return None
    return min(values), max(values), all(v > 0 for v in values)


def fmt(values, decimals=0):
    """Format a list of values as 'median (from lo to hi)', using the middle half."""
    med, lo, hi, n = summarize(values)
    if n == 0:
        return "n/a"
    d = decimals
    return f"{med:.{d}f} (from {lo:.{d}f} to {hi:.{d}f})"


def compare_legs(label, left_values, right_values, unit):
    """Return a sentence if the legs clearly differ, else None.

    "Clearly" = the middle halves of the two distributions do not overlap.
    """
    ml, lo_l, hi_l, nl = summarize(left_values)
    mr, lo_r, hi_r, nr = summarize(right_values)
    if nl == 0 or nr == 0:
        return None
    if max(lo_l, lo_r) <= min(hi_l, hi_r):
        return None
    side = "right" if mr > ml else "left"
    return (f"{label}: higher on the {side} "
            f"({max(ml, mr):.0f} vs {min(ml, mr):.0f}{unit if unit.startswith('°') else ' ' + unit}).")


def filming_tips(recording="treadmill"):
    """Return the filming tips for a recording type, as a list of '- ' lines."""
    if recording == "following":
        return [
            "- Keep the whole body in frame and the camera as steady and level as you can, "
            "at about hip height, looking at the runner from the side.",
            "- Keep a constant distance from the runner.",
            "- 120 frames per second or more (slow-motion mode on a phone).",
            f"- At least {MIN_DURATION_S:.0f}-15 seconds of continuous running at a steady speed, "
            "tight-fitting clothing, good light.",
            "- These tips for a moving camera are suggestions of this project, not from the "
            "article, which describes a fixed camera.",
        ]
    return [
        "- Fixed camera, at a fixed angle perpendicular to the running direction.",
        "- 120 frames per second or more (slow-motion mode on a phone).",
        "- Whole body always in frame, tight-fitting clothing, good light.",
        "- On a treadmill: run 6-10 minutes before filming, at the speed of a long run.",
        f"- At least {MIN_DURATION_S:.0f}-15 seconds of continuous running.",
    ]


def build_report(fps, duration, speed_kmh, cadence, osc, trunk, legs, recording="treadmill"):
    """Return (main_sections, background_sections), each a list of (title, lines).

    Lines starting with "- " are bullet points; any other line is an explanatory note.
    The main part is short and in plain language: what to do first, what was measured, what
    to check alone. The background part holds the numbers, the limits and the research.
    `legs` is {"left": leg_dict, "right": leg_dict}. Only measures that are reliable
    enough are shown as exact results; at low frame rates the foot landing position
    is shown as a range, and the tibia angle is left out. `recording` is "treadmill"
    or "following" (outdoors, camera following the runner).

    The report describes measurements and what the research says. It never tells a
    runner without a problem to change cadence, foot strike or anything else.
    """
    following = recording == "following"
    sections = []   # background sections

    speed = f"{speed_kmh:g} km/h" if speed_kmh else "not provided"
    rec_label = "outdoors, camera following the runner" if following else "treadmill"
    result_lines = [
        f"Video: {duration:.1f} s at {fps:.0f} fps | running speed: {speed} | recording: {rec_label}",
        "These are measurements, not a diagnosis. The studies give no cut-off values between "
        "good and bad running form.",
    ]

    high_fps = fps >= MIN_FPS_FOR_EVENT_MEASURES
    usable = {n: leg for n, leg in legs.items() if leg["strikes_used"] >= MIN_STRIKES_PER_LEG}

    # Limits of this particular recording
    limits = []
    if following:
        limits.append("Outdoor recording with a moving camera: this mode has not been tested on real "
                      "videos. Camera movement can affect the detection of foot strikes, and a camera "
                      "that tilts or changes distance can distort the angles.")
        limits.append("The vertical oscillation of the pelvis is not reported, because the vertical "
                      "movement of the camera would add to the runner's own.")
    if duration < MIN_DURATION_S:
        limits.append(f"The video is shorter than {MIN_DURATION_S:.0f} seconds: few steps, "
                      "less stable measures.")
    if not high_fps:
        limits.append(f"Only {fps:.0f} frames per second (one frame = {1000 / fps:.0f} ms): "
                      "foot position at landing is given only as a range and the tibia "
                      "angle cannot be measured reliably.")
    for name, leg in legs.items():
        if name not in usable:
            limits.append(f"{name.capitalize()} leg: only {leg['strikes_used']} usable strikes "
                          f"(at least {MIN_STRIKES_PER_LEG} are needed), so it is not summarized.")
    sections.append(("HOW TO READ THE NUMBERS", [
        "'from X to Y' is the middle half of the strikes, i.e. how much the measure varies from "
        "step to step.",
    ]))

    # Whole body
    body = []
    if not np.isnan(cadence):
        body.append(f"- Cadence: {cadence:.0f} steps/min (what the research says about it is "
                    "in a section further down).")
    if not following and summarize(osc)[3]:
        body.append(f"- Vertical oscillation of the pelvis: {fmt(osc, 1)} % of leg length. "
                    "According to the article, a larger vertical excursion is associated with "
                    "higher impact and braking forces. There are no cut-off values: it is mainly "
                    "useful for comparing videos filmed the same way.")
    med, lo, hi, n = summarize(trunk)
    if n:
        body.append(f"- Trunk lean: {med:.0f}° (positive = leaning forward; from {lo:.0f} to {hi:.0f}). "
                    "It also depends on how level the camera is; according to the article, the "
                    "evidence on which trunk lean is preferable is still limited.")
    if body:
        sections.append(("WHOLE BODY", body))

    # Knee
    if usable:
        lines = [f"- {name.capitalize()} leg: at touchdown {fmt(leg['flex_touchdown'])}°, "
                 f"peak in stance {fmt(leg['peak_flexion'])}°" for name, leg in usable.items()]
        lines.append("  The article states that peak knee flexion in stance approaches about 45° on "
                     "average and that much lower values may suggest reduced shock absorption, but "
                     "it gives no cut-offs. Estimates from 2D video are approximate.")
        sections.append(("KNEE (flexion: 0° = straight leg)", lines))

    # Foot position at touchdown: always shown, but at low frame rates only as a range
    if usable:
        lines = []
        for name, leg in usable.items():
            if high_fps:
                lines.append(f"- {name.capitalize()} leg: ankle relative to pelvis {fmt(leg['landing'])} % "
                             f"of leg length (positive = in front); tibia angle {fmt(leg['tibia'])}° "
                             f"(positive = knee in front of the ankle)")
                continue
            summary = landing_summary(leg)
            if summary is None:
                continue
            lo, hi, ahead = summary
            if ahead:
                verdict = "in every reading the ankle is in front of the pelvis centre"
            else:
                verdict = ("the ankle is not always in front of the pelvis centre: "
                           "even the direction depends on the frame read")
            lines.append(f"- {name.capitalize()} leg: between {lo:.0f} and {hi:.0f} % of leg length, "
                         f"depending on the frame read; {verdict}.")

        if high_fps:
            lines.append(f"  Read {loading_delay_frames(fps)} frames after touchdown.")
        else:
            lines.append("  At this frame rate the exact value depends on the frame chosen (the ankle "
                         "moves backwards relative to the body during stance), so it is given as a "
                         "range between 1 and 3 frames after touchdown. The tibia angle is not shown "
                         "because it is not stable with this recording.")
        lines.append("  The reference is the midpoint between the two hips; the article instead asks "
                     "whether the line through the ankle bone falls within the pelvis, and the width "
                     "of the pelvis cannot be measured in this view: a small positive value may "
                     "therefore still be 'under the pelvis'. The article also warns that this measure "
                     "ignores trunk lean and works less well for midfoot and forefoot strikers, and "
                     "that it should be judged together with other measures.")
        sections.append(("FOOT POSITION AT LANDING (read together with the other measures)", lines))

    # Left / right differences
    if len(usable) < 2:
        lines = ["- Not comparable: enough valid strikes are needed for both legs."]
    else:
        pairs = [("Knee flexion at touchdown", "flex_touchdown", "°"),
                 ("Peak knee flexion in stance", "peak_flexion", "°")]
        if high_fps:
            pairs += [("Ankle relative to pelvis", "landing", "% of leg length"),
                      ("Tibia angle", "tibia", "°")]
        found = [compare_legs(label, usable["left"][key], usable["right"][key], unit)
                 for label, key, unit in pairs]
        found = [s for s in found if s]
        if found:
            lines = [f"- {sentence}" for sentence in found]
            lines.append("  A difference may be real or due to the recording: the leg closest to the "
                         "camera is tracked better and perspective distorts the measures. To find out, "
                         "repeat the video filmed from the opposite side: if the difference follows "
                         "the leg closest to the camera, it is probably a recording effect.")
        else:
            lines = ["- No clear difference: the middle halves of the left and right values overlap."]
    sections.append(("DIFFERENCES BETWEEN LEGS", lines))

    # What the research says: descriptive statements only, no advice to change anything
    if not np.isnan(cadence):
        sections.append(("WHAT THE RESEARCH SAYS ABOUT CADENCE", [
            "- When runners raised their step rate at the same speed, several biomechanical "
            "factors associated with running injuries, such as impact loads, decreased (a "
            "systematic review of 10 studies, almost all with few participants).",
            "- Whether this prevents injuries has not been established. A 2025 review of 18 "
            "studies describes cadence retraining as a promising, low-cost strategy that may "
            "reduce overuse injuries, while a meta-analysis of 37 studies found only 2 studies "
            "on injury and judged the evidence insufficient on both injury and performance.",
            "- For running economy, a higher step frequency was only weakly associated with "
            "better economy (meta-analysis of observational studies).",
            "- The roughly 180 steps/min often quoted comes mostly from running-economy studies "
            "rather than injury studies, and there is no cadence that is ideal for everyone. "
            "Running speed affects running mechanics, so compare videos recorded at the same speed.",
        ]))

    # What the report does not recommend: always shown
    sections.append(("WHAT THIS REPORT DOES NOT RECOMMEND", [
        "- This report does not tell you that your running form is wrong: no study gives cut-off "
        "values. The ideas in 'check it yourself' are experiments backed by small studies, not "
        "corrections, and none of them has been shown to prevent injuries.",
        "- A 5-10% increase in cadence appears in the research as part of supervised clinical "
        "protocols for runners with patellofemoral (kneecap) pain, where a review of reviews "
        "concluded that it may be a feasible complementary intervention, with feedback that is "
        "gradually reduced. It is not presented as a recommendation for runners without a problem.",
        "- This report does not classify your foot strike. For uninjured runners who land on the "
        "heel, a meta-analysis (Anderson et al., 2020) concluded that changing foot strike cannot "
        "be recommended: the relationship between foot strike and injury could not be "
        "determined, and the load shifts towards the ankle and calf. A systematic review (Burke "
        "et al., 2021) found low evidence of a relationship between foot strike and injuries.",
        "- If you have pain or an injury, a qualified professional (for example a physiotherapist "
        "or a sports doctor) can assess you and, if appropriate, supervise changes to your "
        "running.",
    ]))

    sections.append(("SOURCES", [
        "- 'An Evidence-Based Videotaped Running Biomechanics Analysis': framework for the "
        "measures in this report.",
        "- Anderson et al. (2020), Sports Medicine: 'What are the Benefits and Risks Associated "
        "with Changing Foot Strike Pattern During Running? A Systematic Review and Meta-analysis "
        "of Injury, Running Economy, and Biomechanics'.",
        "- Burke et al. (2021): 'Risk Factors for Injuries in Runners: A Systematic Review of "
        "Foot Strike Technique and Its Classification at Impact'.",
        "- Figueiredo et al. (2025), Cureus: 'The Influence of Running Cadence on Biomechanics "
        "and Injury Prevention: A Systematic Review'.",
        "- 'Influence of stride frequency and length on running mechanics: a systematic review', "
        "Sports Health (2014).",
        "- 'What is the effect of changing running step rate on injury, performance and "
        "biomechanics?' (systematic review and meta-analysis).",
        "- 'Gait retraining for patellofemoral pain in runners: an umbrella review of clinical "
        "and biomechanical evidence'.",
        "- 'The relationship between running biomechanics and running economy: a systematic "
        "review and meta-analysis of observational studies', Sports Medicine (2024).",
    ]))

    # ----------------------------- main part: simple ------------------------------
    legs_differ = []
    if len(usable) == 2 and high_fps:
        pairs = [("Knee flexion at touchdown", "flex_touchdown", "°"),
                 ("Peak knee flexion in stance", "peak_flexion", "°")]
        if high_fps:
            pairs += [("Ankle relative to pelvis", "landing", "% of leg length"),
                      ("Tibia angle", "tibia", "°")]
        legs_differ = [x for x in (compare_legs(label, usable["left"][key],
                                                usable["right"][key], unit)
                                   for label, key, unit in pairs) if x]

    items = []
    if legs_differ:
        items.append(Advice(
            COACH, "Your two legs do not move the same way",
            saw=" ".join(legs_differ),
            try_this="Film yourself again from the other side. If the difference stays with the "
                     "same leg, it may be real: pay attention to it while you run, and if you "
                     "have pain on one side, ask a physiotherapist.",
            why="Experts who review running videos check that the right and left sides move the "
                "same way, because a difference can point to weakness or tightness on one side. "
                "But the leg closest to the camera is tracked better and perspective changes the "
                "numbers, so a difference can come from the video, and no study gives a limit "
                "for how big a difference is a problem.",
            sure="low", priority=5,
            source="Vincent et al. 2014 (symmetry is part of a running video check); no reference "
                   "value for how big a difference matters."))

    # Step rate: compared with a lab group running at about the same speed
    ref = reference_cadence(speed_kmh)
    low_cadence = (ref is not None and not np.isnan(cadence)
                   and cadence < ref[0] - LOW_CADENCE_SD * ref[1])
    faster = int(round(cadence * (1 + STEP_RATE_CHANGE))) if not np.isnan(cadence) else None
    metronome = ("Use a metronome app or a watch that beeps at the new rate, and run with it two "
                 "or three times a week for about two weeks. Shorten your steps a little instead "
                 "of running faster. Stop if anything hurts. At first it may feel awkward and a "
                 "bit harder.")
    cadence_why = ("In studies, running with 5 to 10% more steps at the same speed lowered the "
                   "work done by the knee when the foot lands and the impact loading. A metronome "
                   "helped runners change their step rate in two weeks without raising their "
                   "heart rate. It has not been shown to prevent injuries, and the best step "
                   "rate is different for everybody.")
    if low_cadence:
        items.append(Advice(
            COACH, "Your steps are on the slow side for your speed",
            saw=f"{cadence:.0f} steps per minute at {speed_kmh:g} km/h. Regular runners in a lab "
                f"study at about this speed took roughly {ref[0]:.0f} (most between "
                f"{ref[0] - ref[1]:.0f} and {ref[0] + ref[1]:.0f}).",
            try_this=f"Try about {faster} steps per minute (5% more than now) for part of a run. "
                     + metronome,
            why=cadence_why, sure="low", priority=4,
            source="Reference: Fukuchi et al. 2017; effects: Heiderscheit et al. 2011, Doyle et "
                   "al. 2022, Huber & Verhoff 2024. The limit (1 SD below the group) is a choice "
                   "of this project."))

    checks = [
        Advice(CHECK, "Run tall, do not sit",
               try_this="Push your hips forward and up, as if a string pulled you up from the "
                        "top of your head, and let your feet land under your body, not far in "
                        "front of it.",
               why="Common coaching advice to feel lighter. The studies read do not test this "
                   "cue, and a video from the side cannot show your hips well enough to judge "
                   "it.",
               source="Common coaching advice."),
    ]
    if faster is not None and not low_cadence:
        checks.append(Advice(
            CHECK, "If you want to try quicker, shorter steps",
            try_this=f"At the same speed, try about {faster} steps per minute (5% more than your "
                     f"{cadence:.0f} now) for part of a run. " + metronome +
                     " This is an experiment, not a target.",
            why=cadence_why,
            source="Heiderscheit et al. 2011; Doyle et al. 2022; Huber & Verhoff 2024."))
    checks.append(Advice(
        CHECK, "Let your arms swing freely",
        try_this="Keep your shoulders relaxed and let your arms swing, with elbows bent, instead "
                 "of holding them still.",
        why="In a lab study, running with the arms held still cost 3 to 13% more energy and made "
            "the trunk twist more.",
        source="Arellano & Kram 2014."))
    checks.append(Advice(
        CHECK, "Land softly and quietly",
        try_this="Listen to your feet and aim for quiet, soft landings. You do not need to change "
                 "how your foot lands.",
        why="Soft-landing cues lowered impact in runners with kneecap pain. Switching from heel to "
            "forefoot landing is not recommended for runners who land on the heel: the load "
            "moves to the ankle and calf.",
        source="Petri et al. 2026; Vincent et al. 2014; Anderson et al. 2020; Burke et al. 2021."))

    main = [("YOUR RESULT", result_lines)]
    main.append(what_to_do_section(
        items, "Nothing in this video clearly breaks a rule backed by research. The studies "
               "give no cut-off values for running form, so I do not tell you to change what "
               "you do. If you have no pain, the ideas below are experiments, not corrections."))
    extra = extra_section(items)
    if extra:
        main.append(extra)

    measured = []
    if not np.isnan(cadence):
        line = f"- Step rate: {cadence:.0f} steps per minute."
        if ref is not None:
            line += (f" For comparison, regular runners in a lab study at about {speed_kmh:g} km/h "
                     f"averaged roughly {ref[0]:.0f}. It is not a target: there is no best "
                     "number.")
        measured.append(line)
    if not following and summarize(osc)[3]:
        measured.append(f"- Up and down movement of the hips: {summarize(osc)[0]:.1f}% of your "
                        "leg length. Runners who bounce less tend to use a little less energy, "
                        "but no value is a target. Useful to compare videos filmed the same way.")
    t_med = summarize(trunk)[0]
    if not np.isnan(t_med):
        measured.append(f"- Trunk lean: {t_med:.0f}° forward (it also depends on how level the "
                        "camera is).")
    for name, leg in usable.items():
        measured.append(f"- {name.capitalize()} knee: bent about "
                        f"{summarize(leg['peak_flexion'])[0]:.0f}° at its most during the "
                        "stance.")
    if measured:
        measured.append("  No advice on these numbers: the studies give no reference values.")
        main.append(("WHAT I MEASURED (NO ADVICE)", measured))

    main.append(check_section(checks))

    trust = [f"- {text}" for text in limits] or ["- No problem found with this video."]
    trust.append("- Angles from a 2D video are repeatable but only moderately close to lab "
                 "measurements, so use them to compare your own videos rather than as exact "
                 "values.")
    if not following:
        trust.append("- Treadmill running is mostly like running on the road. The main differences "
                     "in the studies: the foot lands about 10° flatter and the body moves up and "
                     "down about 1.5 cm less.")
    trust.append("- Running injuries also depend on things a video cannot show, such as how much "
                 "and how fast you increase your training, previous injuries and your general "
                 "health.")
    main.append(("HOW MUCH TO TRUST THIS", trust))
    if limits:
        main.append(("FILMING TIPS", filming_tips(recording)))

    sections.insert(len(sections) - 1, ("WHAT THE RESEARCH ADDS", [
        "- Step rate by speed (Fukuchi et al. 2017, 28 regular runners, 3D lab): 80.8, 85.7 and "
        "91.7 strides per minute at 9, 12.6 and 16.2 km/h, that is about 162, 171 and 183 steps "
        "per minute. Values in between are interpolated here. The group differs from you in "
        "many ways: it is a reference to compare with, not a target.",
        "- A study of 29 collegiate cross-country runners found a step rate of 177 steps per "
        "minute on average, and injured runners (175) did not differ clearly from the others "
        "(179); heavier runners tended to take fewer steps.",
        "- Raising the step rate 5 to 10% reduced energy absorbed at the knee (5%) and hip "
        "(10%), step length, vertical movement of the body and braking (Heiderscheit et al. "
        "2011), and raised the activity of the gluteal muscles in late swing (Chumanov et al. "
        "2012). A 2022 review of 19 trials found, with moderate certainty, that step-rate "
        "retraining raised step rate and lowered the average loading rate, and two trials "
        "reported fewer injuries after one year. Another review found that short-term effects "
        "on injury and performance are still unclear and that a higher step rate felt harder at "
        "first.",
        "- A two-week programme with a metronome (5 to 10% above the starting rate, 2-3 runs a "
        "week) raised the step rate of 33 novice and recreational runners without changing "
        "heart rate, effort or distance; the biomechanics did not change (Huber & Verhoff "
        "2024). A review of low-cost tools found strong evidence that metronomes, smartwatches "
        "and cameras work for changing step rate and foot strike (Dorst et al. 2023).",
        "- Soft-landing cues and small step-rate increases are the most consistent parts of gait "
        "retraining for runners with kneecap pain (umbrella review, Petri et al. 2026); results "
        "for foot strike changes and for injury prevention were less consistent.",
        "- Running economy: a smaller vertical displacement (r = 0.35), a slightly higher "
        "cadence (r = -0.20) and stiffer legs went with a lower energy cost; most angles and "
        "contact times showed no clear link (Van Hooren et al. 2024). Arm swing lowers the "
        "energy cost compared with holding the arms still (Arellano & Kram 2014).",
        "- Treadmill and road running: most measures are similar, but on the treadmill the foot "
        "lands about 10° flatter, the knee bends 6° less from landing to its peak, and the "
        "pelvis moves 1.5 cm less vertically (Van Hooren et al. 2020).",
        "- Quality of the measure: 2D video angles repeat well but agree only moderately with 3D "
        "motion capture, and worst for the frontal and transverse planes (2023 review). "
        "Averages of fewer than 10 strides can be misleading for small effects (Fox et al. "
        "2023).",
        "- Injury risk factors: an umbrella review of 13 low-quality reviews found that training "
        "characteristics, health and lifestyle, and morphological and biomechanical factors "
        "are all linked with running injuries (Correia et al. 2024).",
        "- Pronation: gait retraining can reduce foot pronation (Mousavi et al. 2024), but it "
        "needs a view from behind, so it is not measured here.",
    ]))
    sections[-1][1].extend([
        "- Fukuchi, Fukuchi and Duarte (2017), PeerJ: 'A public dataset of running biomechanics and "
        "the effects of running speed on lower extremity kinematics and kinetics'.",
        "- Luedke and Rauh (2021), Frontiers in Sports and Active Living: 'Factors associated with "
        "self-selected step rates between collegiate and high school cross country runners'.",
        "- Heiderscheit et al. (2011), Medicine & Science in Sports & Exercise: 'Effects of step "
        "rate manipulation on joint mechanics during running'.",
        "- Chumanov et al. (2012), Gait & Posture: 'Changes in muscle activation patterns when "
        "running step rate is increased'.",
        "- Doyle et al. (2022), JOSPT: 'The effectiveness of gait retraining on running "
        "kinematics, kinetics, performance, pain, and injury in distance runners'.",
        "- Huber and Verhoff (2024), IJSPT: 'A one session gait retraining protocol with "
        "metronome augmentation increases cadence in novice and recreational runners'.",
        "- Dorst et al. (2023), Applied Sciences: 'Effectiveness of lower-cost strategies for "
        "running gait retraining'.",
        "- Petri et al. (2026), Sports Medicine - Open: 'Gait retraining for patellofemoral pain "
        "in runners: an umbrella review'.",
        "- Mousavi et al. (2024), PLOS ONE: 'Gait retraining targeting foot pronation'.",
        "- Arellano and Kram (2014), Journal of Experimental Biology: 'The metabolic cost of "
        "human running: is swinging the arms worth it?' (abstract).",
        "- Van Hooren et al. (2020), Sports Medicine: 'Is motorized treadmill running "
        "biomechanically comparable to overground running?'.",
        "- Fox et al. (2023), PeerJ: 'Measurement error associated with gait cycle selection in "
        "treadmill running at various speeds'.",
        "- Vincent et al. (2014), Current Sports Medicine Reports: 'Setting standards for "
        "medically-based running analysis'.",
        "- Correia et al. (2024), Journal of Sport and Health Science: 'Risk factors for "
        "running-related injuries: an umbrella systematic review'.",
        "- 2D video validity and reliability of joint angles in running: systematic review and "
        "meta-analysis (2023, abstract).",
        "- Ferber et al. (2024), Scientific Data: open dataset of 1,798 healthy and injured "
        "runners and walkers (raw data; not used for any value in this report).",
    ])
    sections[-1][1].extend(source_lines(items + checks))
    return main, sections


def report_to_text(sections):
    """Format the sections of build_report() as plain text (for the terminal or a .txt file)."""
    bar = "=" * 64
    lines = []
    for i, (title, body) in enumerate(sections):
        if i == 0:
            lines += [bar, title, bar]
        else:
            lines += ["", title]
        lines += body
    return "\n".join(lines)


# ----------------------------------- the analysis -------------------------------

def analyze_video(video, speed_kmh=None, output_dir="data/output", recording="treadmill"):
    """Run the full running analysis on one video.

    `recording` is "treadmill" (camera still) or "following" (outdoors, camera
    following the runner).

    Returns a dict with:
      sections       the main report (plain language), as a list of (title, lines)
      background     the numbers, limits and research, in the same form
      report_text    the same report as plain text
      details        the technical numbers, as text
      skeleton_path  path of the video with the skeleton drawn on it
      fps, duration  properties of the video

    Raises FileNotFoundError or ValueError (with a readable message) if the video
    cannot be analyzed.
    """
    if recording not in RECORDING_OPTIONS.values():
        raise ValueError(f"Unknown recording type: {recording}")
    following = recording == "following"

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    # Everything printed here is kept as the "technical details" text
    details = io.StringIO()
    with contextlib.redirect_stdout(details):
        fps, points = extract_landmarks(video)
        if len(points) == 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not read any frame from the video.")
        duration = len(points) / fps
        if duration < 2:
            raise ValueError("The video is too short: at least a few seconds of running are needed.")
        detected = float(np.mean(~np.isnan(points).all(axis=(1, 2))))
        if detected < 0.5:
            raise ValueError(f"The runner was detected in only {detected:.0%} of the frames. "
                             "Make sure the whole body is visible and the lighting is good.")

        left_td, left_off = detect_strikes(points, fps, "l")
        right_td, right_off = detect_strikes(points, fps, "r")
        direction = running_direction(points)

        print(f"Video: {duration:.1f}s at {fps:.0f} fps")
        print(f"Recording: {'outdoors, camera following the runner' if following else 'treadmill'}")
        print(f"Running direction: towards the {'right' if direction == 1 else 'left'} of the image")
        print(f"Runner detected in {detected:.0%} of the frames")
        print(f"Foot strikes: {len(left_td)} left, {len(right_td)} right")

        # Whole body
        print("\nWhole body")
        cadence, used, discarded = cadence_from_strides([left_td, right_td], fps)
        if used:
            print(f"  Cadence: {cadence:.0f} steps/min (from {used} strides, {discarded} discarded)")
        else:
            print("  Too few foot strikes detected to compute cadence.")

        osc = []
        if following:
            # The camera's own vertical movement would add to the pelvis movement
            print("  Vertical oscillation: not computed (moving camera)")
        else:
            mean_length = float(np.nanmean([leg_length(points, "l"), leg_length(points, "r")]))
            all_td = np.sort(np.concatenate([left_td, right_td]))
            if len(all_td) >= 2:
                osc = vertical_oscillation(points, fps, all_td, mean_length)
                report("Vertical oscillation of the pelvis", osc, "% of leg length",
                       decimals=1, what="steps")
        trunk = trunk_lean_series(points, direction)
        report("Trunk lean (forward = positive)", trunk, "deg", what="frames")

        # Per leg
        legs = {
            "left": analyze_leg(points, fps, "l", left_td, left_off, direction),
            "right": analyze_leg(points, fps, "r", right_td, right_off, direction),
        }
        print_details("Left", legs["left"])
        print_details("Right", legs["right"])

    # Video with the skeleton and the touchdown frames marked. OpenCV writes a format that
    # browsers often cannot play, so it is re-encoded as H.264 when possible
    out_dir = Path(output_dir)
    raw_path = out_dir / f"{video.stem}_skeleton_raw.mp4"
    final_path = out_dir / f"{video.stem}_skeleton.mp4"
    render_skeleton(video, raw_path, fps, points, left_td, right_td)
    if to_h264(raw_path, final_path):
        raw_path.unlink()
    else:
        raw_path.replace(final_path)

    main_sections, background = build_report(
        fps, duration, speed_kmh, cadence if used else float("nan"), osc, trunk, legs,
        recording=recording)
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
    """Command-line use: python -m src.sports.running VIDEO [SPEED_KMH] [treadmill|following]"""
    args = sys.argv[1:]
    video = Path(args[0]) if args else Path("data/input/run.mp4")
    speed_kmh = None
    if len(args) > 1:
        try:
            speed_kmh = float(args[1].replace(",", "."))
        except ValueError:
            print(f"The speed must be a number in km/h, got: {args[1]}")
            return
    recording = args[2] if len(args) > 2 else "treadmill"
    if recording not in RECORDING_OPTIONS.values():
        print(f"The recording type must be one of: {', '.join(RECORDING_OPTIONS.values())}")
        return

    print("Analyzing the video (this can take a minute)...")
    try:
        result = analyze_video(video, speed_kmh, recording=recording)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    print(result["details"])
    print(f"Skeleton video saved to: {result['skeleton_path']}\n")
    print(result["report_text"])


if __name__ == "__main__":
    main()