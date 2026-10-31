# Squat analysis

Back to the [general README](../../README.md) · code: [`src/sports/squat.py`](../../src/sports/squat.py) · checks: [`tests/test_gym_synthetic.py`](../../tests/test_gym_synthetic.py)

Upload a video of yourself squatting, filmed **from the side**. The report counts your squats,
measures how deep you go and how your chest and shins lean, and gives short tips in plain
language.

> **Status: new, not tested on real videos yet.** The code was checked only on simulated
> stick figures with known angles (see [Testing](#testing)). Whether the pose model follows a
> real lifter well enough, and whether the limits below are sensible, still has to be checked.

<!--
Add a frame of your own video here, then remove the comment markers:
![Video with the detected skeleton and a ring on the hip at the lowest point of each squat](skeleton.png)
-->

## How to film: side view

```
              camera on a tripod
              (hip height, 2.5-3 m away)
                       |
                       v
       lifter: ONE SIDE of the body faces the camera
```

- Camera **beside you**, at about hip height, 2.5 to 3 metres away, looking straight at your side:
  not from the front, not at an angle.
- In the app, choose which side of your body faces the camera (left or right). Only that side is
  analysed and drawn.
- Your whole body in the picture, from head to feet, including the bar. Do not cut off your feet.
- Tight-fitting clothes and good light. A bar on the shoulders can hide them: check that the
  skeleton still follows your body.
- A whole set: at least 3 squats (5 or more is better), without moving the camera.
- 60 frames per second or more if your phone allows it.

These camera positions are suggestions of this project, not values from the studies.

## What the report gives

The report starts with **what to do**: at most three tips, most important first, each with what
the video showed, what to try, why, and how sure the report is. Every tip has one of three labels:

| Label | Meaning | Tips in the squat report |
|---|---|---|
| **backed by research** | measured in your video and supported by a study | Your squat works mostly your hips / mostly your thighs |
| **coach's tip** | measured in your video, but common coaching advice that the studies we read did not test | Your heels come off the floor; you stop above thigh-level |
| **check it yourself** | the camera cannot see it | Knees in line with your toes; feet about shoulder-width apart with toes turned out a little; keep your back in its natural shape |

The "check it yourself" tips are shown to everyone. The measured tips appear only when the
squats in the video are similar to each other and clearly outside the limits below. If nothing
stands out, the report says so.

## What is measured

At the lowest point of each squat (median of a few frames around it):

| Measure | How |
|---|---|
| Knee bend (0° = straight leg) | angle at the knee between hip and ankle |
| Chest lean from upright | line from hip to shoulder compared with the vertical |
| Shin lean from upright | line from ankle to knee compared with the vertical |
| Hips or knees | chest lean minus shin lean |
| Heel rise | how much the heel rises compared with the toe, as a share of foot length |

## Limits and where they come from

| Rule | Limit | Source |
|---|---|---|
| Hips do more of the work | chest lean minus shin lean above +10° | Straub & Powers 2024 (working definition, based on Barrack et al.) |
| Knees do more of the work | chest lean minus shin lean below -10° | same |
| Partial / thigh-level / deep squat | knee bend below 90° / 90-110° / above 110° | Straub & Powers 2024 (working definitions) |
| Heels counted as raised | 15% of foot length or more | **assumption** of this project |
| A squat is counted | hip goes down by at least 20% of leg length, knee bends at least 40°, at least 1 s between squats | **assumptions** |
| A tip is given | at least 3 squats; middle half of the squats within 20° | **assumptions** |

## Notes on the evidence

- Straub & Powers (2024) is a descriptive review (level of evidence 5, the lowest on their
  scale). Lorenzetti et al. (2018) studied healthy people with no or light extra weight. Neither
  followed people over time, so neither shows that a technique prevents or causes an injury.
- **"Don't let your knees come in on the way up, or you will activate the adductors."** The
  rule "keep your knees in line with your toes" is a general guideline, and the report gives it
  as a "check it yourself" tip. The explanation about the adductors is **not** in the papers we
  have: Straub & Powers report more adductor activity with a wide stance on the way up and with
  the toes turned out 30-50°, and Lorenzetti et al. list too much adductor activity as one
  possible *cause* of knees falling inwards, not as an effect. So the report does not use that
  explanation.
- In Lorenzetti et al., knees falling inwards were rare in healthy people (seen with a wide stance),
  and beginners' knees tended to be further out than those of experienced squatters. That is why
  the report does not present "knees caving in" as the typical mistake.
- Heels: the studies show that raised heels let the shin lean further forward and raise the load
  at the knee. They do not say that raised heels are wrong, so "keep your heels down" is labelled
  as a coach's tip.

## What is not measured

- Whether the knees drift inwards or outwards (a front view would be needed).
- The width of the stance and the angle of the feet.
- Rounding of the lower back.
- The weight on the bar, the position of the bar and the speed of the lift.
- The side turned away from the camera.

## Testing

`tests/test_gym_synthetic.py` builds a stick figure with known angles, adds noise and missing
frames, and checks that the analysis finds the same angles back, counts the squats, and gives
(or does not give) the expected tips. It checks the maths and the rules only. It does **not**
check how well the pose model tracks a real person.

```powershell
python -m tests.test_gym_synthetic
```

To validate with real videos: film yourself from the side, check that the skeleton follows your
body, and compare the angles in a few frames with a manual measurement (for example with Kinovea).

## Command line

```powershell
python -m src.sports.squat data\input\squat_side.mp4 right
```

The last word is the side of your body that faces the camera: `left` or `right`.

## Sources

- Straub and Powers (2024), *International Journal of Sports Physical Therapy*: A Biomechanical
  Review of the Squat Exercise: Implications for Clinical Practice
  ([doi:10.26603/001c.94600](https://doi.org/10.26603/001c.94600)), full text read.
- Lorenzetti et al. (2018), *BMC Sports Science, Medicine and Rehabilitation*: How to squat?
  Effects of various stance widths, foot placement angles and level of experience on knee, hip
  and trunk motion and loading
  ([doi:10.1186/s13102-018-0103-7](https://doi.org/10.1186/s13102-018-0103-7)), full text read.