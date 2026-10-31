# Bench press analysis

Back to the [general README](../../README.md) · code: [`src/sports/bench_press.py`](../../src/sports/bench_press.py) · checks: [`tests/test_gym_synthetic.py`](../../tests/test_gym_synthetic.py)

Upload a video of yourself doing the flat bench press, filmed **from the foot end of the bench
with the camera raised**. The report counts your repetitions, measures how wide your hands are
compared with your shoulders and whether the bar stays level, and gives short tips in plain
language.

> **Status: new and the least tested sport.** The code was checked only on simulated stick
> figures (see [Testing](#testing)). The pose model finds lying people less reliably than
> standing ones, and this camera view has not been tried on real videos. Most of the important
> bench press points (shoulder blades, bar path) **cannot** be seen by the camera and are given
> as "check it yourself" tips.

<!--
Add a frame of your own video here, then remove the comment markers:
![Video with the detected skeleton and a ring on each wrist at the lowest point of each repetition](skeleton.png)
-->

## How to film: from the foot end, camera raised

```
        head -> towards the rack
   [ rack ] [ lifter lying on the bench ]  <----- 3-4 m ----- camera
                                                  (raised, tilted down)
```

- Camera at the **foot end of the bench**, in line with the bench and with you, 3 to 4 metres away.
  Your head is towards the rack.
- Raised to about the height of your chest when standing (1.3 to 1.6 metres) and tilted down, so
  that it sees your chest, both arms and both hands.
- Your shoulders, elbows, wrists and the bar stay in the picture during the whole lift. Your
  knees and the rack must not hide them.
- Tight-fitting top, good light, nothing between the camera and your arms.
- A whole set: at least 3 repetitions (5 or more is better), without moving the camera.
- Use a spotter or the safety arms of the rack for heavy sets, and keep the camera out of their
  way.

This camera position is a suggestion of this project and has not been tested yet.

**Why this view and not the side?** The main thing the research can tell us about from a video
is how wide the hands are compared with the shoulders, and that is only visible along the bench.
The side view would show the bar path, but none of the papers we read gives a reference for it.

## What the report gives

The report starts with **what to do**: at most three tips, most important first, each with what
the video showed, what to try, why, and how sure the report is. Every tip has one of three labels:

| Label | Meaning | Tips in the bench press report |
|---|---|---|
| **backed by research** | measured in your video and supported by a study | Your hands look wide / narrow on the bar |
| **coach's tip** | measured in your video, but common coaching advice that the studies we read did not test | One hand ends higher than the other |
| **check it yourself** | the camera cannot see it | Shoulder blades back and down; try pushing your hands outwards, as if bending the bar |

The "check it yourself" tips are shown to everyone. The measured tips appear only when the
repetitions in the video are similar and clearly outside the limits below.

## What is measured

| Measure | How |
|---|---|
| Repetitions | movement of the wrists compared with the shoulders |
| Hand width | distance between the wrists divided by the distance between the shoulders, at the lowest point |
| Bar tilt | angle of the line between the wrists compared with the line between the shoulders, at the top |
| Pace | time between two lowest points |

## Limits and where they come from

| Rule | Limit | Source |
|---|---|---|
| Hands wide | hand width 1.75 times the shoulder width or more | **assumption**, placed between the 1.5 and 2 times tested by Noteboom et al. 2024 |
| Hands narrow | 1.2 times the shoulder width or less | **assumption** |
| Bar tilt | at least 5° in at least 70% of the repetitions | **assumption** |
| A repetition is counted | wrists move by at least 0.25 shoulder widths, at least 1 s between repetitions | **assumptions** |
| A tip is given | at least 3 repetitions; hand width within 0.3 in the middle half | **assumptions** |

The pose model gives the shoulder joints, not the bone tips (the width between the bone tips is
what the studies used), and the wrists, not the exact point where the hand holds the bar. So the
hand width is only approximate, and the report gives advice only if it is clearly wide or narrow.

## Notes on the evidence

- **Noteboom et al. (2024):** 10 experienced lifters (9 men, 1 woman), a light 16 kg bar, loads
  estimated by a computer model of the shoulder. A grip narrower than 1.5 times the width between
  the shoulder bones gave less pressure on the collarbone joint and less backward push on the
  shoulder joint; pulling the shoulder blades back gave less load on the shoulder joint and less
  work for the rotator cuff. The angle of the upper arm had a smaller effect. It is not a study of
  injuries.
- **Larsen et al. (2021):** recreationally trained men, maximal lift. Wide and medium grips lifted
  more than a narrow one. This is why the "hands wide" tip is a choice (lift more, or ease the
  shoulders) and not a rule.
- **"Bend the bar":** in a simulation of one lift, pushing outwards on the bar lowered the shoulder
  loads. The authors say the numbers cannot be generalised, and it was not tested as a tip with
  real lifters. The two least experienced lifters pushed inwards. The report gives it as an idea to
  try, in the "check it yourself" list, not as a measured tip.
- **Schütz et al. (2022):** 20 healthy people, moderate loads. The bench press had lower and
  steadier shoulder loads than cable cross-overs. It does not compare bench press techniques, so
  it is not used for the tips.
- None of these studies followed people over time.

## What is not measured

- The shoulder blades.
- The path of the bar and where it touches the chest.
- How far the elbows point out from the body (the view along the bench squashes it; the study
  found it had a smaller effect than grip width and shoulder blades).
- The push of the hands inwards or outwards on the bar.
- The weight on the bar and the speed of the bar.

## Testing

`tests/test_gym_synthetic.py` builds a stick figure for the foot-end view with a known hand width
and bar tilt, adds noise and missing frames, and checks that the analysis counts the repetitions,
finds the values back and gives (or does not give) the expected tips. It checks the maths and the
rules only. It does **not** check how well the pose model tracks a lying person.

```powershell
python -m tests.test_gym_synthetic
```

To validate with real videos: film yourself from the foot end, check that the skeleton follows
your shoulders and wrists in every repetition, and compare the hand width in a few frames with a
manual measurement.

## Command line

```powershell
python -m src.sports.bench_press data\input\bench_foot_end.mp4
```

## Sources

- Noteboom et al. (2024), *Frontiers in Physiology*: Effects of bench press technique variations
  on musculoskeletal shoulder loads and potential injury risk
  ([doi:10.3389/fphys.2024.1393235](https://doi.org/10.3389/fphys.2024.1393235)), full text read.
- Larsen, Gomo and van den Tillaar (2021), *Frontiers in Sports and Active Living*: A
  Biomechanical Analysis of Wide, Medium, and Narrow Grip Width Effects on Kinematics, Horizontal
  Kinetics, and Muscle Activity on the Sticking Region in Recreationally Trained Males During
  1-RM Bench Pressing
  ([doi:10.3389/fspor.2020.637066](https://doi.org/10.3389/fspor.2020.637066)), full text read.
- Schütz et al. (2022), *Sports*: Chest Exercises: Movement and Loading of Shoulder, Elbow and
  Wrist Joints ([doi:10.3390/sports10020019](https://doi.org/10.3390/sports10020019)), full text
  read.