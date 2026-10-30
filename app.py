"""Streamlit interface: upload a video, choose the sport, read the analysis report."""

import base64
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.sports import cycling, running

# One entry per supported sport: name shown in the interface -> module with analyze_video()
SPORTS = {"Running": running, "Cycling": cycling}

st.set_page_config(page_title="AI Coaching", layout="centered")


def show_video(video_bytes):
    """Show a video by embedding it in the page as a data URI.

    This avoids Streamlit's media file server, which raised an ASGI error here.
    """
    data = base64.b64encode(video_bytes).decode("ascii")
    components.html(
        '<video controls style="width:100%;max-height:460px;object-fit:contain;'
        f'border-radius:8px" src="data:video/mp4;base64,{data}"></video>',
        height=480,
    )


def show_report(sections):
    """Render the report sections: bullets as a list, other lines as small notes."""
    for title, lines in sections:
        st.subheader(title.capitalize())
        bullets = []
        for line in lines:
            if line.startswith("- "):
                bullets.append(line)
                continue
            if bullets:
                st.markdown("\n".join(bullets))
                bullets = []
            st.caption(line.strip())
        if bullets:
            st.markdown("\n".join(bullets))


st.title("AI Coaching")
st.write("Upload a video of yourself and get an analysis of your technique.")
st.info("This tool gives measurements and observations, not a medical diagnosis. "
        "If you have pain or an injury, talk to a health professional.")

sport_name = st.selectbox("Sport", list(SPORTS))
sport = SPORTS[sport_name]

# Extra question of the sport (how the video was filmed, or which side faces the camera)
recording_options = getattr(sport, "RECORDING_OPTIONS", {})
recording = None
if recording_options:
    question = getattr(sport, "OPTION_QUESTION", "How was the video filmed?")
    recording_label = st.radio(question, list(recording_options))
    recording = recording_options[recording_label]
    if recording == "following":
        st.warning("Outdoor videos with a moving camera have not been tested yet: treat the "
                   "results with extra caution. The vertical oscillation is not reported "
                   "in this mode.")

tips = sport.filming_tips(recording) if hasattr(sport, "filming_tips") else []
if tips:
    with st.expander(f"How to film yourself ({sport_name.lower()})"):
        st.markdown("\n".join(tips))

uploaded = st.file_uploader("Video", type=["mp4", "mov", "m4v", "avi"])

# The running speed is only asked for sports that use it
speed = 0.0
if hasattr(sport, "SPEED_HELP"):
    speed_help = sport.SPEED_HELP.get(recording, "Leave at 0 if you do not know it.")
    speed = st.number_input(
        "Running speed (km/h)", min_value=0.0, max_value=40.0, value=0.0, step=0.5,
        help=speed_help + " For now it is only written in the report.",
    )

if st.button("Analyse", type="primary", disabled=uploaded is None):
    # OpenCV reads from a file path, so the upload is saved to a temporary folder
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video_path = Path(tmp) / f"upload{suffix}"
        video_path.write_bytes(uploaded.getbuffer())
        options = {"recording": recording} if recording else {}
        with st.spinner("Analysing the video (this can take a minute)..."):
            try:
                result = sport.analyze_video(video_path, speed or None,
                                             output_dir=Path(tmp), **options)
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
                st.stop()
            # Read the video before the temporary folder is deleted
            result["skeleton_bytes"] = Path(result["skeleton_path"]).read_bytes()
    # Keep the result: Streamlit re-runs this script at every click
    st.session_state["result"] = result
    st.session_state["result_sport"] = sport_name

# Show the result only for the sport it was computed for
result = st.session_state.get("result")
if result and st.session_state.get("result_sport") == sport_name:
    st.divider()
    st.subheader("Video with the detected skeleton")
    st.caption("Check that the skeleton follows your body. If it does not, the "
               "measurements below are not reliable. " + getattr(sport, "VIDEO_NOTE", ""))
    show_video(result["skeleton_bytes"])
    st.download_button("Download the video with the skeleton", result["skeleton_bytes"],
                       file_name="skeleton.mp4", mime="video/mp4")

    show_report(result["sections"])

    with st.expander("Technical details"):
        st.code(result["details"], language=None)
    st.download_button("Download the report (.txt)", result["report_text"],
                       file_name=f"{sport_name.lower()}_report.txt")