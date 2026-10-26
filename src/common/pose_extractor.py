"""Extract body landmarks from a video using MediaPipe Pose."""

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe landmark indices for the body parts that matter in running
LANDMARKS = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_toe": 31, "r_toe": 32,
}
NAMES = list(LANDMARKS.keys())


def extract_landmarks(video_path):
    """Return (fps, points) where points has shape [n_frames, n_landmarks, 2].

    Coordinates are in pixels. Frames with no detected person are filled with NaN.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frames = []
    with mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # MediaPipe expects RGB, OpenCV gives BGR
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            row = np.full((len(NAMES), 2), np.nan)
            if result.pose_landmarks:
                for i, idx in enumerate(LANDMARKS.values()):
                    lm = result.pose_landmarks.landmark[idx]
                    # Convert normalized coordinates to pixels so angles are not distorted
                    row[i] = (lm.x * width, lm.y * height)
            frames.append(row)

    cap.release()
    return fps, np.array(frames)