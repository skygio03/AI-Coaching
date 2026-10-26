# AI-Coaching

AI-powered sports technique analysis. Upload a video of yourself, get measurements of your
movement and observations you can discuss with a coach or a health professional.

> **Status: early prototype.** Only running (side view) is implemented. It has been tested on
> very few videos and has not been validated against a reference system, so treat every number
> as approximate.

**[Try it online](https://ai-coaching-awyt3ladz87argtgwttf3k.streamlit.app/)**

## What the result looks like

The app draws the detected skeleton on your video, so you can check that the tracking is right
before trusting any number. A ring marks each foot touchdown found by the analysis (L = left
foot, R = right foot).

![Video with the detected skeleton and foot touchdown markers](docs/images/skeleton-example.png)

The report that follows lists the measures, how much they vary from step to step, and which
ones could not be measured reliably with that recording.

![The web interface, showing the video and the beginning of the report](docs/images/app-screenshot.png)

*Example from a 30 fps video. If the skeleton does not follow the body, the measurements
below it are not reliable.*

## How it works

1. **Pose estimation** with [MediaPipe](https://developers.google.com/mediapipe): body landmarks
   (shoulders, elbows, wrists, hips, knees, ankles, heels, toes) in every frame.
2. **Event detection**: foot touchdown and toe-off for each leg.
3. **Biomechanical measures** computed from the landmarks: joint angles, foot position relative
   to the pelvis, vertical oscillation, cadence.
4. **A plain-language report** that shows only the measures that are reliable enough for the
   recording, and says which ones are not.

The measures follow the article
[*An Evidence-Based Videotaped Running Biomechanics Analysis*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4714754/).
That article gives **no cut-off values** between normal and abnormal running, so this project
does not label anything as "wrong". It reports measurements, how much they vary from step to
step, and how reliable they are.

## Running analysis: what is measured

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

Open the address shown in the terminal (usually http://localhost:8501), choose the sport, upload
a video, enter the running speed if you know it, and press **Analyse**. The page shows the video
with the detected skeleton, the report, and the technical details.

Command line:

```powershell
python -m src.sports.running data\input\video.mp4 14
```

The last number is the running speed in km/h (optional). For now it is only written in the
report: the article notes that speed changes running kinematics, so it is useful for comparing
videos recorded at the same speed.

## Filming tips

- Side view: camera still, at a fixed angle perpendicular to the running direction.
- Whole body always in frame, tight-fitting clothing, good light.
- 120 frames per second or more if possible (slow-motion mode on a phone).
- At least 10-15 seconds of continuous running at a steady speed.
- On a treadmill: run 6-10 minutes before filming, at the speed of a long run.

## Project structure

```
app.py                    Streamlit interface
requirements.txt
docs/images/              screenshots used in this README
src/
  common/                 shared by all sports
    pose_extractor.py     MediaPipe landmarks from a video
    render.py             skeleton drawing and browser-friendly video encoding
  sports/
    running.py            running analysis and report
data/                     your videos and outputs (not tracked by git)
```

To add a sport: create `src/sports/<sport>.py` with an `analyze_video(video, speed_kmh,
output_dir)` function that returns the same dictionary as `running.analyze_video`, then add the
module to the `SPORTS` dictionary in `app.py`.

## Limitations

- This is not a medical device and gives no diagnosis. If you have pain or an injury, talk to a
  health professional.
- 2D video from a single camera: perspective and camera position affect the measures.
- Only the sagittal plane (side view) is analysed; nothing is measured from behind or in front.
- The thresholds used by the report (minimum frame rate, minimum number of steps, ...) are
  choices of this project, not values from the article; they are marked as assumptions at the
  top of `src/sports/running.py`.
- Videos are processed locally and deleted from the temporary folder when the analysis ends. If
  you ever host this app online, uploaded videos will contain people: decide what is stored and
  write a privacy notice first.