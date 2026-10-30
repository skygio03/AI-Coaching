"""Draw the detected skeleton on a video and save the result."""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.common.pose_extractor import NAMES

# Pairs of landmarks joined by a line
EDGES = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_ankle", "l_heel"), ("l_heel", "l_toe"), ("l_ankle", "l_toe"),
    ("r_ankle", "r_heel"), ("r_heel", "r_toe"), ("r_ankle", "r_toe"),
]

# Colors are BGR (OpenCV order): left side blue, right side orange
LEFT_COLOR = (255, 128, 0)
RIGHT_COLOR = (0, 128, 255)
CENTER_COLOR = (200, 200, 200)


def _edge_color(a, b):
    if a.startswith("l_") and b.startswith("l_"):
        return LEFT_COLOR
    if a.startswith("r_") and b.startswith("r_"):
        return RIGHT_COLOR
    return CENTER_COLOR


def _valid(p):
    """True if the landmark was detected (no NaN in its coordinates)."""
    return not np.isnan(p).any()


def _on_side(name, side):
    """True if a landmark belongs to `side` ('l' or 'r'); always True when side is None."""
    return side is None or name.startswith(f"{side}_")


def render_skeleton(video_path, out_path, fps, points, strikes_left, strikes_right,
                    flash_frames=6, only_side=None):
    """Write a copy of the video with the skeleton drawn on every frame.

    A ring is drawn around the ankle for `flash_frames` frames after each detected event.
    If `only_side` is 'l' or 'r', only the landmarks and bones of that side are drawn
    (the other side, the nose and the lines joining the two sides are left out).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    index = {name: i for i, name in enumerate(NAMES)}

    i = 0
    while i < len(points):
        ok, frame = cap.read()
        if not ok:
            break

        # Create the writer on the first frame, using the real frame size
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        row = points[i]

        # Bones
        for a, b in EDGES:
            if not (_on_side(a, only_side) and _on_side(b, only_side)):
                continue
            pa, pb = row[index[a]], row[index[b]]
            if _valid(pa) and _valid(pb):
                cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)),
                         _edge_color(a, b), 3)

        # Joints
        for name, k in index.items():
            if only_side is not None and not _on_side(name, only_side):
                continue
            if _valid(row[k]):
                cv2.circle(frame, tuple(row[k].astype(int)), 5, (255, 255, 255), -1)

        # Ring around the ankle for a few frames after each event
        for strikes, ankle, color, label, prefix in (
            (strikes_left, "l_ankle", LEFT_COLOR, "L", "l"),
            (strikes_right, "r_ankle", RIGHT_COLOR, "R", "r"),
        ):
            if only_side is not None and prefix != only_side:
                continue
            if any(0 <= i - s < flash_frames for s in strikes) and _valid(row[index[ankle]]):
                center = tuple(row[index[ankle]].astype(int))
                cv2.circle(frame, center, 22, color, 3)
                cv2.putText(frame, label, (center[0] + 26, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        writer.write(frame)
        i += 1

    cap.release()
    if writer is not None:
        writer.release()


def to_h264(src_path, dst_path):
    """Re-encode a video as H.264 so that web browsers can play it. Return True on success.

    OpenCV writes MPEG-4 Part 2 ("mp4v"), which most browsers cannot play.
    The ffmpeg binary comes from the imageio-ffmpeg package.
    """
    try:
        import imageio_ffmpeg

        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src_path),
             # H.264 with yuv420p needs even width and height
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-crf", "23", "-an", str(dst_path)],
            capture_output=True,
        )
        return result.returncode == 0 and Path(dst_path).is_file()
    except Exception:
        return False