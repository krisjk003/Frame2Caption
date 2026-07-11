import json
from pathlib import Path

import streamlit as st

from app import main

st.set_page_config(
    page_title="Frame2Caption",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Frame2Caption")
st.caption("AI-Powered Multi-Style Video Captioning")

video_url = st.text_input(
    "Video URL",
    placeholder="https://....mp4 or any public video URL",
)

styles = st.multiselect(
    "Caption Styles",
    [
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    ],
    default=[
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    ],
)

if st.button("Generate Captions", use_container_width=True):

    if not video_url.strip():
        st.error("Please enter a video URL.")
        st.stop()

    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    task = [
        {
            "task_id": "demo",
            "video_url": video_url,
            "styles": styles,
        }
    ]

    with open("input/tasks.json", "w") as f:
        json.dump(task, f, indent=2)

    with st.spinner("Generating captions..."):

        exit_code = main()

    if exit_code != 0:
        st.error("Caption generation failed.")
        st.stop()

    with open("output/results.json") as f:
        results = json.load(f)

    result = results[0]

    if "error" in result:
        st.error(result["error"])
        st.stop()

    captions = result["captions"]

    c1, c2 = st.columns(2)

    with c1:

        if "formal" in captions:
            st.subheader("Formal")
            st.success(captions["formal"])

        if "humorous_tech" in captions:
            st.subheader("Humorous-Tech")
            st.info(captions["humorous_tech"])

    with c2:

        if "sarcastic" in captions:
            st.subheader("Sarcastic")
            st.warning(captions["sarcastic"])

        if "humorous_non_tech" in captions:
            st.subheader("Humorous-Non-Tech")
            st.success(captions["humorous_non_tech"])

    st.download_button(
        "Download JSON",
        json.dumps(result, indent=2),
        file_name="captions.json",
        mime="application/json",
    )