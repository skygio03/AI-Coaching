"""Cycling position analysis (right side only, static camera, bike on a trainer or rollers)."""

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from src.common.pose_extractor import extract_landmarks, NAMES
from src.common.render import render_skeleton, to_h264
# Helpers shared with the running analysis
from src.sports.running import (fill_gaps, fmt, joint_angle, report_to_text,
                                running_direction, smooth, summarize)

# ---------------------------------------------------------------------------
# Settings. Values marked "assumption" are my own choices, NOT taken from the
# papers; check them against your videos.
# ---------------------------------------------------------------------------
SIDE = "right"        # only the rider's right side is analysed (the camera is on the right)
PREFIX = "r"          # landmark prefix of that side

MIN_REVOLUTIONS = 10              # assumption: a consensus statement gives "mean of ten cycles"
                                  # only as an example of how many cycles to analyse
MIN_DURATION_S = 10.0             # assumption
CADENCE_RANGE_RPM = (40, 150)     # assumption: plausible pedalling rates on a trainer
RING_DURATION_S = 0.3             # assumption: how long the ring stays on the video

# Knee flexion at the bottom of the stroke (dynamic), from three different sources.
# They depend on intensity and on the goal, and must NOT be merged into one range.
KNEE_RANGES = (
    ("2019 review, low intensity", (33, 43)),
    ("2019 review, high intensity", (30, 40)),
    ("2024 review, depending on power or efficiency priority", (25, 35)),
)
# Assumption: a 2024 consensus statement cites a mean bias of about 11 deg in 2D knee flexion
# between absolute and normalised angles, so smaller gaps cannot be told from measurement differences
KNEE_TOLERANCE_DEG = 10
KNEE_MAX_SPREAD_DEG = 10          # assumption: if the middle half of the knee angles is wider, no advice

# Saddle height advice (all assumptions, see the explanation given with the code)
TARGET_KNEE_FLEXION = 35          # assumption: middle of the reported ranges
AVERAGE_LEG_CM = 85               # assumption: thigh + shank of an average adult
MAX_SADDLE_STEP_MM = 10           # assumption: studies found trivial to small effects under 4% of leg length

# Interface: a single option, shown so that the user knows how to film
RECORDING_OPTIONS = {
    "My right side faces the camera (the only side supported for now)": "right",
}
OPTION_QUESTION = "Side of your body facing the camera"
VIDEO_NOTE = "A ring marks the bottom of each pedal stroke (lowest point of the right ankle)."


# ------------------------------- helpers -------------------------------------

def landmark(points, name):
    """Pixel positions of one landmark of the right side, shape [n_frames, 2]."""
    return points[:, NAMES.index(f"{PREFIX}_{name}")]


def angle_series(a, b, c):
    """Interior angle at b (degrees) for every frame; NaN where a point is missing."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if not np.isnan([a[i], b[i], c[i]]).any():
            out[i] = joint_angle(a[i], b[i], c[i])
    return out


def trunk_angle_series(points, direction):
    """Trunk angle from the horizontal for every frame (90 = upright, 0 = flat on the bike).

    Uses the line from the hip to the shoulder of the right side. A straight line
    ignores the curvature of the spine.
    """
    shoulder = landmark(points, "shoulder")
    hip = landmark(points, "hip")
    dx = direction * (shoulder[:, 0] - hip[:, 0])   # positive = shoulder ahead of the hip
    dy = hip[:, 1] - shoulder[:, 1]                 # positive = shoulder above the hip
    return np.degrees(np.arctan2(dy, dx))


def leg_length_px(points):
    """Median thigh + shank length of the right leg, in pixels."""
    hip, knee, ankle = landmark(points, "hip"), landmark(points, "knee"), landmark(points, "ankle")
    lengths = np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)
    return float(np.nanmedian(lengths))


def reach_series(points, direction):
    """Horizontal distance from hip to wrist, as % of leg length (positive = hand ahead of hip).

    Seen from the side, the wrist is used as a stand-in for the hand on the handlebar.
    """
    hip, wrist = landmark(points, "hip"), landmark(points, "wrist")
    return direction * (wrist[:, 0] - hip[:, 0]) / leg_length_px(points) * 100


# ------------------------------ pedal revolutions ------------------------------

def detect_bdc(points, fps):
    """Frames where the right ankle is at its lowest point: bottom dead centre (BDC) proxy.

    In image coordinates y grows downward, so the lowest point is a maximum of y.
    The ankle is not exactly at the crank position, so this is an approximation of the
    6 o'clock method, not the crank angle itself.
    """
    y = smooth(fill_gaps(landmark(points, "ankle")[:, 1]), fps)
    spread = np.percentile(y, 95) - np.percentile(y, 5)
    # Peaks must be far enough apart (about 200 rpm at most) and clearly above the noise
    peaks, _ = find_peaks(y, distance=max(1, int(0.3 * fps)), prominence=0.3 * spread)
    return peaks


def revolutions(bdc, fps):
    """Pairs of consecutive BDC frames that are one plausible pedal revolution apart."""
    shortest = 60.0 / CADENCE_RANGE_RPM[1]
    longest = 60.0 / CADENCE_RANGE_RPM[0]
    return [(int(a), int(b)) for a, b in zip(bdc[:-1], bdc[1:])
            if shortest <= (b - a) / fps <= longest]


def analyze_revolutions(points, fps, pairs):
    """Measure the leg angles in every pedal revolution (from one BDC to the next)."""
    hip = landmark(points, "hip")
    knee = landmark(points, "knee")
    ankle = landmark(points, "ankle")
    toe = landmark(points, "toe")
    shoulder = landmark(points, "shoulder")

    knee_flexion = 180 - angle_series(hip, knee, ankle)   # 0 deg = straight leg
    hip_angle = angle_series(shoulder, hip, knee)         # trunk to thigh, 180 = straight
    ankle_angle = angle_series(knee, ankle, toe)          # shank to foot line, larger = more plantar flexion

    out = {k: [] for k in ("flex_bdc", "max_ext", "knee_range", "hip_min",
                           "ankle_bdc", "ankle_range", "rpm")}
    for a, b in pairs:
        seg = knee_flexion[a:b + 1]
        out["flex_bdc"].append(float(knee_flexion[a]))
        out["ankle_bdc"].append(float(ankle_angle[a]))
        if np.isnan(seg).all():
            out["max_ext"].append(float("nan"))
            out["knee_range"].append(float("nan"))
        else:
            out["max_ext"].append(float(np.nanmin(seg)))                    # most extended knee
            out["knee_range"].append(float(np.nanmax(seg) - np.nanmin(seg)))
        hseg = hip_angle[a:b + 1]
        out["hip_min"].append(float("nan") if np.isnan(hseg).all() else float(np.nanmin(hseg)))
        aseg = ankle_angle[a:b + 1]
        out["ankle_range"].append(float("nan") if np.isnan(aseg).all()
                                  else float(np.nanmax(aseg) - np.nanmin(aseg)))
        out["rpm"].append(60.0 * fps / (b - a))
    return out


# ------------------------------ saddle height advice ------------------------------

def range_gap(value, bounds):
    """Distance in degrees from `value` to the range `bounds` (0 if it is inside)."""
    low, high = bounds
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def saddle_change_mm(flexion_now):
    """Approximate saddle height change, in mm, to bring the knee flexion at BDC to the target.

    Positive = raise the saddle, negative = lower it. Geometry of a leg with two segments of
    equal length: the hip-ankle distance is leg_length * sin(interior_knee_angle / 2). It is an
    approximation for an average adult and ignores the ankle and the angle of the seat tube.
    """
    def hip_ankle_distance(flexion):
        return AVERAGE_LEG_CM * math.sin(math.radians(180 - flexion) / 2)
    return (hip_ankle_distance(TARGET_KNEE_FLEXION) - hip_ankle_distance(flexion_now)) * 10


def round_to_5(mm):
    """Round to the nearest 5 mm, never below 5 mm."""
    return max(5, int(round(mm / 5.0)) * 5)


def saddle_height_lines(flex_values, enough):
    """Plain-language advice on saddle height from the knee angle at the bottom of the stroke."""
    median, q25, q75, n = summarize(flex_values)
    if not enough or n == 0 or np.isnan(median):
        return ["- I could not measure your knee angle reliably in this video, so I cannot say "
                "anything about your saddle height. Film again following the tips above the video."]
    spread = q75 - q25
    if spread > KNEE_MAX_SPREAD_DEG:
        return [f"- Your knee angle changes too much from one pedal stroke to the next in this "
                f"video (by about {spread:.0f}°), so I cannot judge your saddle height. Film "
                "again following the tips above the video."]

    low_edge = min(low for _, (low, _) in KNEE_RANGES)
    high_edge = max(high for _, (_, high) in KNEE_RANGES)
    nearest = min(range_gap(median, rng) for _, rng in KNEE_RANGES)
    bent = (f"At the bottom of the pedal stroke your knee is bent about {median:.0f}° "
            "(0° would be a straight leg).")

    if nearest == 0:
        return [f"- Saddle height looks fine. {bent} That is inside the range reported by the "
                f"studies (about {low_edge}-{high_edge}°, depending on the source and on how hard "
                "you ride). Keep it as it is."]
    if nearest <= KNEE_TOLERANCE_DEG:
        return [f"- Saddle height is probably fine. {bent} That is a little outside the usual "
                f"range (by {nearest:.0f}°), but a video measurement is not precise enough to be "
                "sure. If you feel no discomfort, keep it as it is."]

    total = round_to_5(abs(saddle_change_mm(median)))
    step = round_to_5(min(total, MAX_SADDLE_STEP_MM))
    if median > high_edge:
        head = "Your saddle is probably too low."
        reason = f"{bent} That is more than the usual range (about {low_edge}-{high_edge}°)."
        verb = "raising"
    else:
        head = "Your saddle is probably too high."
        reason = (f"{bent} That is less than the usual range (about {low_edge}-{high_edge}°): "
                  "your leg is almost straight.")
        verb = "lowering"
    lines = [f"- {head} {reason}",
             f"  What to do: try {verb} the saddle by about {step} mm ({step / 10:.1f} cm), ride for "
             "a few minutes and film yourself again."]
    if total > step:
        lines.append(f"  You may need about {total} mm ({total / 10:.1f} cm) in total: do it in "
                     f"steps of about {step} mm and film again after each step.")
    lines.append("  Change only the saddle, one step at a time. If you feel pain in your knee, stop "
                 "adjusting and ask a physiotherapist or a bike fitter.")
    return lines


# ---------------------------------- the report ----------------------------------

def filming_tips(recording=None):
    """Return the filming tips for cycling, as a list of '- ' lines."""
    return [
        "- Put the camera on the RIGHT side of the bike, so that the right side of your body "
        "faces the camera. Only the right side is analysed for now.",
        "- Fixed camera, perpendicular to the bike, so that the whole bike and the whole rider "
        "are always in frame.",
        "- Put the camera on a tripod at about hip height and do not move it during the recording. "
        "Use the same camera position when you want to compare two videos.",
        "- Warm up and pedal for at least 2 minutes at the effort and cadence you want to analyse "
        "before recording, and keep them steady: a 2024 consensus statement recommends this and "
        "notes that exercise intensity, cadence and fatigue influence pedalling kinematics.",
        "- For hard efforts, record at least 30 seconds (same consensus statement).",
        "- 60 frames per second or more if possible (slow-motion mode on a phone).",
        f"- At least {MIN_DURATION_S:.0f}-15 seconds of continuous pedalling, tight-fitting "
        "clothing, shoes and wrists visible, good light.",
        "- Camera side, camera height, 60 fps and clothing are suggestions of this project, "
        "not taken from the papers.",
    ]


def build_report(fps, duration, measures, trunk, reach, elbow):
    """Return (main_sections, background_sections), each a list of (title, lines).

    Lines starting with "- " are bullet points; any other line is an explanatory note.
    The main part gives plain-language results and advice. The background part holds the
    technical numbers, the limits and the research, for who wants to know more.
    """
    n_rev = len(measures["rpm"])
    enough = n_rev >= MIN_REVOLUTIONS
    rpm_median = summarize(measures["rpm"])[0]
    knee_median = summarize(measures["flex_bdc"])[0]

    # ----------------------------- main part: simple ------------------------------
    main = [("YOUR RESULT", [
        f"Video: {duration:.1f} s at {fps:.0f} fps, right side of the rider facing the camera.",
        "These are estimates from a single 2D video: not a medical opinion and not a "
        "professional bike fit.",
    ])]
    main.append(("SADDLE HEIGHT", saddle_height_lines(measures["flex_bdc"], enough)))

    if enough:
        lines = []
        r_med = summarize(reach)[0]
        if not np.isnan(r_med):
            lines.append(f"- Hips to hands: your hands are about {r_med:.0f}% of your leg length in "
                         "front of your hips. I cannot tell you whether that is too long or too "
                         "short: no scientific reference exists for this distance, and I do not "
                         "want to invent one.")
        a_med = summarize(measures["ankle_range"])[0]
        if not np.isnan(a_med):
            lines.append(f"- Ankle: your ankle moves through about {a_med:.0f}° in each pedal "
                         "stroke. There is no reference value for this either, so there is no "
                         "advice. It is useful to compare two videos of yourself: if you change "
                         "your saddle, this number can change too.")
        t_med = summarize(trunk)[0]
        if not np.isnan(t_med):
            lines.append(f"- Back: your back is about {t_med:.0f}° from the horizontal (90° would be "
                         "sitting upright, 0° completely flat). There is no 'right' angle: studies "
                         "disagree, and it depends on whether you care more about comfort, power "
                         "or speed.")
        if lines:
            main.append(("HANDS, ANKLE AND BACK (measured, no advice yet)", lines))

    main.append(("IF YOU FEEL DISCOMFORT WHEN YOU RIDE", [
        "- Lower back: in three small studies, a lower handlebar made the lower back bend more. "
        "Raising the handlebar may reduce that. Studies also link low back pain to weak core and "
        "back muscles, which a video cannot show.",
        "- Knee pain: studies found no clear link between bike position and knee pain, and the "
        "suggestions differ with the type of pain. See a physiotherapist rather than adjusting by "
        "yourself.",
        "- Always change one thing at a time, in small steps, and film again.",
    ]))

    reliab = [f"- {n_rev} pedal strokes were analysed (at least {MIN_REVOLUTIONS} are needed)."]
    if duration < MIN_DURATION_S:
        reliab.append(f"- The video is shorter than {MIN_DURATION_S:.0f} seconds: film a longer one.")
    reliab.append("- Only your right leg is analysed, so the camera must be on the right of the bike.")
    reliab.append("- A knee angle measured from a video can be off by several degrees (a 2024 "
                  "consensus statement cites about 11° between two ways of measuring it). That is "
                  "why I give advice only when your knee angle is clearly outside the usual range.")
    main.append(("HOW MUCH TO TRUST THIS", reliab))

    # --------------------------- background part: details ---------------------------
    bg = []

    # Numbers
    lines = []
    if enough:
        lines += [
            f"- Pedalling rate: {fmt(measures['rpm'])} rpm (from {n_rev} revolutions).",
            f"- Knee flexion at the bottom of the pedal stroke (0° = straight leg): "
            f"{fmt(measures['flex_bdc'])}°",
            f"- Maximum knee extension in each revolution (smallest flexion): "
            f"{fmt(measures['max_ext'])}°",
            f"- Range of knee motion in each revolution: {fmt(measures['knee_range'])}°",
            f"- Ankle angle at the bottom of the stroke (shank to foot line; larger = more "
            f"plantar flexion): {fmt(measures['ankle_bdc'])}°",
            f"- Ankle range of motion in each revolution: {fmt(measures['ankle_range'])}°",
            f"- Hip angle (trunk to thigh), most closed point of each revolution: "
            f"{fmt(measures['hip_min'])}°",
        ]
        t_med, t_lo, t_hi, t_n = summarize(trunk)
        if t_n:
            lines.append(f"- Trunk angle from the horizontal: {t_med:.0f}° (from {t_lo:.0f} to {t_hi:.0f})")
        r_med, r_lo, r_hi, r_n = summarize(reach)
        if r_n:
            lines.append(f"- Horizontal hip-to-wrist distance: {r_med:.0f}% of leg length "
                         f"(from {r_lo:.0f} to {r_hi:.0f})")
        e_med, e_lo, e_hi, e_n = summarize(elbow)
        if e_n:
            lines.append(f"- Elbow angle (180° = straight arm): {e_med:.0f}° (from {e_lo:.0f} to {e_hi:.0f})")
        lines.append("  'from X to Y' is the middle half of the pedal revolutions, i.e. how much the "
                     "measure varies from one revolution to the next.")
    else:
        lines.append(f"- Only {n_rev} usable pedal revolutions were found (at least {MIN_REVOLUTIONS} "
                     "are needed), so the numbers are not reported. Check that the right side of "
                     "the rider is the one facing the camera.")
    bg.append(("THE NUMBERS", lines))

    # Knee comparison with each source
    if enough and not np.isnan(knee_median):
        lines = []
        for label, (low, high) in KNEE_RANGES:
            gap = range_gap(knee_median, (low, high))
            if gap == 0:
                lines.append(f"- Inside the {low}-{high}° range ({label}).")
            else:
                side = "above" if knee_median > high else "below"
                lines.append(f"- {gap:.0f}° {side} the {low}-{high}° range ({label}).")
        lines.append("  The ranges are kept separate because they come from different sources, "
                     "intensities and goals: they differ by up to 8° at each end. The advice above "
                     "is given only when the value is outside all of them by more than "
                     f"{KNEE_TOLERANCE_DEG}°, and only when the knee angle is stable from one "
                     "revolution to the next.")
        lines.append(f"  How the saddle change is computed: the target is about {TARGET_KNEE_FLEXION}° "
                     "(a choice of this project, in the middle of the ranges). The millimetres come "
                     f"from the geometry of a leg of {AVERAGE_LEG_CM} cm (thigh + shank), so they are "
                     "an approximation that is less accurate for very tall or very short riders. "
                     f"A single step is limited to {MAX_SADDLE_STEP_MM} mm because a 2022 review "
                     "(abstract read) found trivial to small effects for changes under 4% of leg "
                     "length, and a 2011 review reports that a 5% change alters the knee kinematics "
                     "a lot.")
        lines.append("  Method: dynamic measurement from video, with the lowest point of the ankle "
                     "as an approximation of the 6 o'clock crank position (the consensus statement "
                     "asks to report the method, because different methods give systematically "
                     "different results). Angle = hip, knee and ankle points estimated by the pose "
                     "model, not skin markers. Knee and hip angles are not uniquely determined by "
                     "the crank position unless the ankle angle is also known (a 2024 review notes "
                     "this); more ankle plantar flexion gives a higher knee flexion angle (2011 "
                     "review).")
        lines.append("  Static values, such as the 25-30° suggested by a 2011 review for the Holmes "
                     "method with the rider seated, are a different method and are not comparable "
                     "with a dynamic video measurement.")
        bg.append(("THE KNEE ANGLE COMPARED WITH THE STUDIES", lines))

    bg.append(("WHAT CANNOT BE MEASURED FROM THIS VIDEO", [
        "- Curvature of the lumbar spine: the back angle only joins shoulder and hip.",
        "- Knee tracking in the frontal plane (a front view would be needed).",
        "- Foot position on the pedal, saddle fore-aft position, saddle tilt, handlebar reach, "
        "crank length and cleat position: the bike itself is not measured.",
        "- Hand position on the handlebar (hoods, drops, grips).",
        "- The left leg: only the right side is analysed.",
    ]))

    bg.append(("WHAT THE RESEARCH SAYS", [
        "- Saddle height: a 2022 systematic review (abstract read) found strong evidence that "
        "saddle height should be set with dynamic knee-angle measurements, moderate evidence that "
        "changes under 4% of leg length have trivial to small effects on lower-limb loads, and "
        "limited evidence on injury risk.",
        "- A 2011 review concluded that there was not enough evidence to define an optimal saddle "
        "height from knee forces or joint kinematics, while suggesting a static knee flexion of "
        "25-30°; the Holmes method itself proposes values between 25° and 35° depending on the "
        "knee condition.",
        "- A 2019 review (abstract read) reports dynamic knee flexion at the bottom of the stroke of "
        "33-43° at low intensity and 30-40° at high intensity, and notes that hip, shoulder and "
        "elbow have not been studied in a meaningful way.",
        "- A 2024 review reports a broad consensus on a dynamic knee flexion at the bottom of the "
        "stroke between 25° and 35°, depending on whether peak power or energy efficiency is the "
        "priority. A flexion of about 25° was associated with higher peak power in two studies, "
        "while findings on energy cost were mixed.",
        "- Knee pain: a 2018 review of studies comparing cyclists with and without knee pain (all "
        "cross-sectional and retrospective) found, in one study, about 2° more knee adduction and "
        "5° more ankle dorsiflexion in cyclists with pain, and no difference in knee flexion angle "
        "or in knee joint forces. A 2017 review of 14 small studies of low to moderate quality "
        "found no direct association between bike positioning and knee injury risk.",
        "- The same 2017 review reports opposite suggestions for different conditions (for example "
        "a higher saddle for some medial knee problems and a lower one after an anterior cruciate "
        "ligament injury), so no single direction ('lower' or 'higher') fits everyone with knee "
        "pain.",
        "- In one study of 20 competitive cyclists summarised in the 2018 review, a saddle height "
        "giving about 30° of knee flexion was the most comfortable, and more knee flexion went "
        "with more fatigue and more pain perception in the front of the thigh and knee.",
        "- Low back pain: a 2017 review of 8 small studies of low to moderate quality (all "
        "participants were men) found that lower handlebar positions increased lumbar flexion in "
        "the three studies that tested handlebar height, and that cyclists with low back pain "
        "showed core muscle activation imbalances, back extensor endurance deficits and more "
        "lumbar flexion. The authors suggest that the time spent in a flexed position and these "
        "muscle factors may matter more than the position alone.",
        "- A 2024 systematic review of studies of competitive and trained cyclists (recreational "
        "cyclists were excluded) analysed 47 studies, none rated good quality (5 fair, 33 poor, "
        "10 not rated). Only 4 studies had more than 25 participants, and most participants were "
        "men (454 men, 90 women, 180 not specified). Consensus exists only for saddle height, "
        "and there are no clear recommendations for crank length, handlebar height, Q factor or "
        "cleat position. Several studies found more power with a more upright trunk, results on "
        "oxygen cost were mixed, and a lower trunk reduces drag, which is not measured here. "
        "The findings may not describe a recreational rider.",
        "- A 2024 Delphi consensus of 14 experts encourages dynamic assessment after a warm-up of "
        "at least 2 minutes at the intended intensity and cadence, asks to report the number of "
        "cycles analysed and the knee-flexion method (6 o'clock method preferred), prefers 3D but "
        "accepts 2D if its limits (joint centres, parallax) are stated, and reached no consensus "
        "on how to define 2D angles.",
        "- Why this app does not give advice on the distance to the hands, the ankle or the "
        "back angle: none of the papers read gives a reference value for them.",
    ]))

    bg.append(("SOURCES", [
        "- Bini, Hume and Croft (2011), Sports Medicine: 'Effects of Bicycle Saddle Height on Knee "
        "Injury Risk and Cycling Performance' (full text read).",
        "- Priego-Quesada et al. (2024), Sports Medicine: 'Bicycle Set-Up Dimensions and Cycling "
        "Kinematics: A Consensus Statement Using Delphi Methodology' (full text read).",
        "- Husband et al. (2024), Journal of Sports Sciences: 'Cycling position optimisation - a "
        "systematic review of the impact of positional changes on biomechanical and physiological "
        "factors in cycling' (full text read).",
        "- Johnston et al. (2017), International Journal of Sports Physical Therapy: 'The influence "
        "of extrinsic factors on knee biomechanics during cycling: a systematic review of the "
        "literature' (full text read).",
        "- Bini and Bini (2018), Open Access Journal of Sports Medicine: 'Potential factors "
        "associated with knee pain in cyclists: a systematic review' (full text read).",
        "- Streisfeld et al. (2017), Sports Health: 'Relationship Between Body Positioning, Muscle "
        "Activity, and Spinal Kinematics in Cyclists With and Without Low Back Pain: A Systematic "
        "Review' (full text read).",
        "- Bini and Priego-Quesada (2022), Journal of Sports Sciences: 'Methods to determine saddle "
        "height in cycling and implications of changes in saddle height in performance and injury "
        "risk: A systematic review' (abstract only).",
        "- Swart and Holliday (2019), Current Sports Medicine Reports: 'Cycling biomechanics "
        "optimization - the (R)evolution of bicycle fitting' (abstract only): source of the "
        "dynamic knee-angle ranges by intensity.",
    ]))

    return main, bg


# ----------------------------------- the analysis -------------------------------

def stat_line(label, values, unit, decimals=0):
    """Print one measurement as: median (middle half of the values, count)."""
    med, lo, hi, n = summarize(values)
    print(f"  {label}: {med:.{decimals}f} {unit} (middle half: {lo:.{decimals}f} to "
          f"{hi:.{decimals}f}, {n} values)")


def analyze_video(video, speed_kmh=None, output_dir="data/output", recording="right"):
    """Run the cycling analysis on one video (right side of the rider facing the camera).

    `recording` must be "right": it is kept so that every sport has the same signature.
    `speed_kmh` is not used for cycling, for the same reason.

    Returns a dictionary like running.analyze_video, plus "background" (the technical part of
    the report). Raises FileNotFoundError or ValueError (with a readable message) if the video
    cannot be analyzed.
    """
    if recording != SIDE:
        raise ValueError("Only the right side of the rider facing the camera is supported for now.")

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
            raise ValueError("The video is too short: at least a few seconds of pedalling are needed.")
        detected = float(np.mean(~np.isnan(points).all(axis=(1, 2))))
        if detected < 0.5:
            raise ValueError(f"The rider was detected in only {detected:.0%} of the frames. "
                             "Make sure the whole body is visible and the lighting is good.")

        direction = running_direction(points)
        bdc = detect_bdc(points, fps)
        pairs = revolutions(bdc, fps)
        measures = analyze_revolutions(points, fps, pairs)
        trunk = trunk_angle_series(points, direction)
        reach = reach_series(points, direction)
        elbow = angle_series(landmark(points, "shoulder"),
                             landmark(points, "elbow"),
                             landmark(points, "wrist"))

        print(f"Video: {duration:.1f}s at {fps:.0f} fps")
        print(f"Analysed side: {SIDE}")
        print(f"Rider facing the {'right' if direction == 1 else 'left'} of the image")
        print(f"Rider detected in {detected:.0%} of the frames")
        print(f"Bottom-of-stroke frames found: {len(bdc)}; usable pedal revolutions: {len(pairs)}")
        stat_line("Pedalling rate", measures["rpm"], "rpm")
        stat_line("Knee flexion at the bottom of the stroke", measures["flex_bdc"], "deg")
        stat_line("Maximum knee extension per revolution", measures["max_ext"], "deg")
        stat_line("Range of knee motion per revolution", measures["knee_range"], "deg")
        stat_line("Ankle angle at the bottom of the stroke", measures["ankle_bdc"], "deg")
        stat_line("Ankle range of motion per revolution", measures["ankle_range"], "deg")
        stat_line("Hip angle, most closed point per revolution", measures["hip_min"], "deg")
        stat_line("Trunk angle from the horizontal", trunk, "deg")
        stat_line("Hip-to-wrist distance", reach, "% of leg length")
        stat_line("Elbow angle", elbow, "deg")

    # Video with only the right side of the skeleton; a ring marks the bottom of each stroke.
    # OpenCV writes a format browsers often cannot play, so it is re-encoded as H.264
    out_dir = Path(output_dir)
    raw_path = out_dir / f"{video.stem}_skeleton_raw.mp4"
    final_path = out_dir / f"{video.stem}_skeleton.mp4"
    none = np.array([], dtype=int)
    ring_frames = max(3, int(round(RING_DURATION_S * fps)))
    render_skeleton(video, raw_path, fps, points, none, bdc,
                    flash_frames=ring_frames, only_side=PREFIX)
    if to_h264(raw_path, final_path):
        raw_path.unlink()
    else:
        raw_path.replace(final_path)

    main_sections, background = build_report(fps, duration, measures, trunk, reach, elbow)
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
    """Command-line use: python -m src.sports.cycling VIDEO"""
    args = sys.argv[1:]
    video = Path(args[0]) if args else Path("data/input/bike_side_static.mp4")

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