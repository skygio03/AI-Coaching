# AI-Coaching

AI-powered sports technique analysis. Upload a video of yourself, get measurements of your
movement and observations you can discuss with a coach or a health professional.

> **Status: early prototype.** Running (side view) and cycling (right side, bike on a trainer or
> rollers) are implemented. Both have been tested on very few videos and have not been validated
> against a reference system, so treat every number as approximate.

## What the result looks like

The app draws the detected skeleton on your video, so you can check that the tracking is right
before trusting any number. In running, a ring marks each foot touchdown found by the analysis
(L = left foot, R = right foot). In cycling, only the right side is drawn and a ring marks the
bottom of each pedal stroke.

![Video with the detected skeleton and foot touchdown markers](docs/images/skeleton-example.png)

The report that follows lists the measures, how much they vary from step to step, and which
ones could not be measured reliably with that recording.

![The web interface, showing the video and the beginning of the report](docs/images/app-screenshot.png)

*Example from a 30 fps running video. If the skeleton does not follow the body, the
measurements below it are not reliable.*

## How it works

1. **Pose estimation** with [MediaPipe](https://developers.google.com/mediapipe): body landmarks
   (shoulders, elbows, wrists, hips, knees, ankles, heels, toes) in every frame.
2. **Event detection**: foot touchdown and toe-off for each leg in running; the bottom of each
   pedal stroke (lowest point of the right ankle) in cycling.
3. **Biomechanical measures** computed from the landmarks: joint angles, foot position relative
   to the pelvis, vertical oscillation, cadence.
4. **A plain-language report** that shows only the measures that are reliable enough for the
   recording, and says which ones are not.

## Running

The measures follow the article
[*An Evidence-Based Videotaped Running Biomechanics Analysis*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4714754/).
That article gives **no cut-off values** between normal and abnormal running, so this project
does not label anything as "wrong". It reports measurements, how much they vary from step to
step, and how reliable they are.

Two ways of filming are supported: treadmill with a still camera, or outdoors with the camera
following the runner. A still camera on a runner who moves across the frame is not supported.

| Measure | Reliability with a 30 fps video |
|---|---|
| Cadence | Good |
| Vertical oscillation of the pelvis | Good |
| Trunk lean | Good (depends on how level the camera is) |
| Knee flexion at touchdown and in stance | Fair |
| Foot landing position relative to the pelvis | Poor: shown only as a range |
| Tibia angle at loading response | Poor: not shown below 100 fps |
| Ground contact time | Not reliable: not reported |

The reliability column comes from testing on a single 30 fps video: the foot landing position and
the tibia angle changed noticeably when the frame they are read at moved by just one frame (33 ms).
Foot landing position and tibia angle are therefore shown as exact values only for videos of 100 fps
or more, which has not been tested yet.

Left/right differences are reported only when the middle halves of the two distributions do not
overlap, and they may be caused by the camera position (the leg closest to the camera is tracked
better) rather than by the runner.

## Cycling

Only the **right side** of the rider is analysed, with a fixed camera on the right of the bike
(bike on a trainer or rollers). Pedal revolutions are found from the lowest point of the right
ankle, used as an approximation of the 6 o'clock crank position.

| Measure | What the report does with it |
|---|---|
| Knee flexion at the bottom of the pedal stroke | **Advice on saddle height**, only when it is clearly outside the ranges of the studies |
| Knee range, ankle angle and range, hip angle | Measured and shown in the details, no advice |
| Hip-to-hands distance, ankle movement, back angle | Measured, **no advice**: none of the papers read gives a reference value |
| Pedalling rate | Measured |

**How the saddle advice is decided.** The knee angle is compared with three ranges that are kept
separate because they come from different sources and goals: 33-43° (2019 review, low intensity),
30-40° (2019 review, high intensity) and 25-35° (2024 review, power vs efficiency). Advice is given
only when the value is outside all of them by more than 10° (a 2024 consensus statement cites
about 11° between two ways of measuring the knee angle) and when the angle is stable from one
revolution to the next. The saddle change in millimetres comes from simple geometry (target 35°,
leg of 85 cm), is limited to 10 mm per step, and the report asks you to film again after each
step. These choices (target angle, step size, tolerance) are assumptions of this project, not
values from the papers.

**What cycling does not do (yet):**

- It has been tested on one video found online, with no way to check the real saddle height. The
  advice has **not** been checked against real saddle changes on the same rider.
- Static knee angles (such as the 25-30° of the Holmes method) are a different method and are
  not comparable with this dynamic measurement from video.
- Studies of freely chosen positions (Holliday & Swart 2021: knee 36° static, hip 77°, shoulder
  112°, elbow 19°, ankle 116°) describe what 50 well-trained men do, not what is optimal, and are
  not used as cut-offs.
- The left leg, the frontal plane, saddle fore-aft, handlebar reach, crank length and cleat
  position are not measured.

## Install (Windows, PowerShell)

Python **3.9 to 3.12** is required: MediaPipe 0.10.14 does not support newer versions.

```powershell
git clone https://github.com/skygio03/AI-Coaching.git
cd AI-Coaching
python3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Use

Web interface:

```powershell
streamlit run app.py --server.maxUploadSize 500
```

Open the address shown in the terminal (usually http://localhost:8501), choose the sport from the
drop-down menu, answer the question about how the video was filmed, upload the video (enter the
running speed if you know it) and press **Analyse**. The page shows the video with the detected
skeleton, the report, and the technical details.

Command line:

```powershell
python -m src.sports.running data\input\video.mp4 14 treadmill
python -m src.sports.cycling data\input\video.mp4
```

For running, the second number is the speed in km/h (optional) and the last word is
`treadmill` or `following`. The speed is only written in the report: speed changes running
kinematics, so it is useful for comparing videos recorded at the same speed.

## Filming tips

**Running**

- Side view: camera at a fixed angle perpendicular to the running direction (still on a
  treadmill, or following the runner).
- Whole body always in frame, tight-fitting clothing, good light.
- 120 frames per second or more if possible (slow-motion mode on a phone).
- At least 10-15 seconds of continuous running at a steady speed.
- On a treadmill: run 6-10 minutes before filming, at the speed of a long run.

**Cycling**

- Camera on the right side of the bike, fixed and perpendicular to it, on a tripod at about hip
  height, with the whole bike and rider in frame.
- Pedal for at least 2 minutes at the effort and cadence you want to analyse before filming,
  then film at least 10-15 seconds of steady pedalling.
- Tight-fitting clothing, good light, 60 fps or more if possible.

## Project structure

```
app.py                    Streamlit interface
requirements.txt
packages.txt              system libraries needed by Streamlit Community Cloud
docs/images/              screenshots used in this README
src/
  common/                 shared by all sports
    pose_extractor.py     MediaPipe landmarks from a video
    render.py             skeleton drawing and browser-friendly video encoding
  sports/
    running.py            running analysis and report
    cycling.py            cycling analysis and report
data/                     your videos and outputs (not tracked by git)
```

To add a sport: create `src/sports/<sport>.py` with an `analyze_video(video, speed_kmh,
output_dir, recording)` function that returns the same dictionary as `running.analyze_video`
(plus the module attributes used by `app.py`, see `cycling.py`), then add the module to the
`SPORTS` dictionary in `app.py`.

## Limitations

- This is not a medical device and gives no diagnosis. If you have pain or an injury, talk to a
  health professional.
- 2D video from a single camera: perspective and camera position affect the measures.
- Only the side view is analysed; nothing is measured from behind or in front.
- The thresholds used by the reports (minimum frame rate, minimum number of steps or
  revolutions, saddle target and step, ...) are choices of this project, not values from the
  papers; they are marked as assumptions at the top of `src/sports/running.py` and
  `src/sports/cycling.py`.
- Videos are processed locally and deleted from the temporary folder when the analysis ends. If
  you ever host this app online, uploaded videos will contain people: decide what is stored and
  write a privacy notice first.