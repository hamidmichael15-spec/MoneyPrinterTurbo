"""Movie Commentary Studio: turn an uploaded clip into a narrated vertical short.

This page intentionally uses the existing MoneyPrinterTurbo LLM configuration for
script generation and Edge TTS for no-key narration. Scene understanding is guided
by user-provided scene notes in this MVP; it does not claim automatic visual analysis.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import textwrap
from pathlib import Path

import edge_tts
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeAudioClip, ImageClip, VideoFileClip, afx

from app.services import llm
from app.utils import utils

MAX_UPLOAD_BYTES = 250 * 1024 * 1024
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "m4v"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]
DEFAULT_VOICE = "en-US-AriaNeural"


def _safe_name(name: str, fallback: str) -> str:
    suffix = Path(name or "").suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else fallback


def _split_sentences(script: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", (script or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _caption_png(text: str, width: int = 960) -> np.ndarray:
    image = Image.new("RGBA", (width, 260), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_path = utils.font_dir()
    font_file = None
    if font_path and os.path.isdir(font_path):
        for candidate in ("BeVietnamPro-Bold.ttf", "MicrosoftYaHeiBold.ttc"):
            candidate_path = os.path.join(font_path, candidate)
            if os.path.exists(candidate_path):
                font_file = candidate_path
                break
    try:
        font = ImageFont.truetype(font_file, 48) if font_file else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = textwrap.wrap(text, width=30)[:3]
    line_heights = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        line_heights.append(box[3] - box[1])
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 12
    y = max(8, (260 - total_height) // 2)
    for line, line_height in zip(lines, line_heights):
        draw.text(
            (width // 2, y), line, font=font, fill="white",
            anchor="ma", stroke_width=3, stroke_fill="black"
        )
        y += line_height + 12
    return np.array(image)


async def _make_tts(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
    await communicate.save(output_path)


def _make_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Fill a 1080x1920 canvas and center-crop without stretching."""
    source_ratio = clip.w / clip.h
    target_ratio = 9 / 16
    if source_ratio > target_ratio:
        resized = clip.resized(height=1920)
        return resized.cropped(
            x_center=resized.w / 2, y_center=resized.h / 2,
            width=1080, height=1920
        )
    resized = clip.resized(width=1080)
    return resized.cropped(
        x_center=resized.w / 2, y_center=resized.h / 2,
        width=1080, height=1920
    )


def _caption_clips(script: str, duration: float) -> list[ImageClip]:
    sentences = _split_sentences(script)
    if not sentences:
        return []
    weights = [max(len(item), 1) for item in sentences]
    total = sum(weights)
    cursor = 0.0
    overlays = []
    for sentence, weight in zip(sentences, weights):
        segment = duration * weight / total
        image = ImageClip(_caption_png(sentence))
        overlays.append(
            image.with_start(cursor).with_duration(segment).with_position(("center", "bottom"))
        )
        cursor += segment
    return overlays


def _render_video(
    source_path: str, narration_path: str, output_path: str,
    script: str, music_path: str | None, original_volume: float,
    music_volume: float,
) -> None:
    source = VideoFileClip(source_path)
    narration = AudioFileClip(narration_path)
    music = None
    final = None
    try:
        video = _make_vertical(source)
        duration = min(float(video.duration), float(narration.duration))
        if duration <= 0:
            raise ValueError("The uploaded clip or generated narration has no usable duration.")
        video = video.subclipped(0, duration)
        narration = narration.subclipped(0, min(narration.duration, duration)).with_volume_scaled(1.0)
        audio_tracks = [narration]
        if source.audio is not None and original_volume > 0:
            original = source.audio.subclipped(0, min(source.duration, duration)).with_volume_scaled(original_volume)
            audio_tracks.append(original)
        if music_path:
            music = AudioFileClip(music_path)
            music = music.subclipped(0, min(music.duration, duration)).with_volume_scaled(music_volume)
            audio_tracks.append(music)
        mixed_audio = CompositeAudioClip(audio_tracks).with_duration(duration)
        overlays = _caption_clips(script, duration)
        final = video.with_audio(mixed_audio)
        if overlays:
            from moviepy import CompositeVideoClip
            final = CompositeVideoClip([final, *overlays], size=(1080, 1920)).with_duration(duration)
        final.write_videofile(
            output_path, codec="libx264", audio_codec="aac",
            fps=30, preset="medium", threads=2, logger=None
        )
    finally:
        if final is not None:
            final.close()
        if music is not None:
            music.close()
        narration.close()
        source.close()


st.set_page_config(page_title="Movie Commentary Studio", page_icon="🎬", layout="wide")
st.title("🎬 Movie Commentary Studio")
st.caption("Turn a real video clip into a narrated, captioned vertical short.")

with st.expander("How this MVP works", expanded=False):
    st.markdown(
        "- Upload a real video clip (up to 250 MB).\n"
        "- Describe the visible action in the scene notes. The current MVP uses these notes "
        "to write commentary; automatic visual scene recognition is not yet included.\n"
        "- Generate narration with the configured MoneyPrinterTurbo language model and "
        "Microsoft Edge TTS (no separate TTS key).\n"
        "- Mix narration, optional background music and optional original clip audio.\n"
        "- Export a 9:16 MP4 with burned-in captions."
    )

uploaded_video = st.file_uploader("1. Upload your video clip", type=VIDEO_TYPES)
scene_notes = st.text_area(
    "2. Describe what happens in the clip",
    placeholder="Example: A man enters a quiet restaurant, notices someone at the table, "
                "and suddenly turns around. Describe only what is visible.",
    height=130,
)
tone = st.selectbox(
    "Commentary style",
    ["Natural storyteller", "Suspenseful", "Funny and engaging", "Dramatic movie recap", "Simple explanation"],
)
language = st.selectbox("Narration language", ["English", "Nigerian English", "French", "Spanish"])
voice_name = st.selectbox(
    "Narrator voice",
    ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural",
     "en-NG-EzinneNeural", "en-NG-AbeoNeural"],
)
with st.expander("Audio controls", expanded=True):
    original_volume = st.slider("Original clip audio", 0.0, 1.0, 0.0, 0.05)
    music_volume = st.slider("Background music volume", 0.0, 0.5, 0.12, 0.01)
    uploaded_music = st.file_uploader("Optional background music", type=AUDIO_TYPES)
    if original_volume == 0:
        st.caption("Original clip audio will be muted.")

if "movie_commentary_script" not in st.session_state:
    st.session_state.movie_commentary_script = ""

col_a, col_b = st.columns(2)
with col_a:
    generate_script = st.button("3. Generate commentary", type="primary", use_container_width=True)
with col_b:
    clear_script = st.button("Clear commentary", use_container_width=True)

if clear_script:
    st.session_state.movie_commentary_script = ""
if generate_script:
    if not scene_notes.strip():
        st.error("Please describe what happens in the clip first.")
    else:
        prompt = (
            f"Write a natural spoken commentary for a short video based only on these scene notes. "
            f"Style: {tone}. Language: {language}. Keep it engaging, accurate, and concise. "
            f"Do not invent names, motives, dialogue, or events that are not in the notes. "
            f"Target about 25 to 35 seconds of speech. Return only the words to be spoken.\n\n"
            f"SCENE NOTES:\n{scene_notes.strip()}"
        )
        with st.spinner("Writing commentary with your configured language model..."):
            result = llm.generate_script(
                video_subject=scene_notes[:500],
                language=language,
                paragraph_number=1,
                video_script_prompt=prompt,
            )
        if result and not result.startswith("Error:"):
            st.session_state.movie_commentary_script = result.strip()
        else:
            st.error("Commentary generation failed. Check the LLM provider and model settings.")

script = st.text_area(
    "4. Review and edit the narration",
    value=st.session_state.movie_commentary_script,
    height=180,
    key="movie_commentary_editor",
    placeholder="Your generated narration will appear here. You can edit it before creating the video.",
)
st.session_state.movie_commentary_script = script

if st.button("5. Create narrated vertical video", type="primary", use_container_width=True):
    if uploaded_video is None:
        st.error("Upload a video clip first.")
    elif not script.strip():
        st.error("Generate or enter commentary first.")
    elif uploaded_video.size > MAX_UPLOAD_BYTES:
        st.error("The video is larger than the 250 MB upload limit.")
    else:
        progress = st.progress(0, text="Preparing your clip...")
        try:
            with tempfile.TemporaryDirectory(prefix="mpt-commentary-") as workdir:
                source_path = os.path.join(workdir, "source" + _safe_name(uploaded_video.name, ".mp4"))
                narration_path = os.path.join(workdir, "narration.mp3")
                output_path = os.path.join(workdir, "movie-commentary.mp4")
                music_path = None
                with open(source_path, "wb") as handle:
                    handle.write(uploaded_video.getbuffer())
                if uploaded_music is not None:
                    if uploaded_music.size > 30 * 1024 * 1024:
                        raise ValueError("Background music must be 30 MB or smaller.")
                    music_path = os.path.join(workdir, "background" + _safe_name(uploaded_music.name, ".mp3"))
                    with open(music_path, "wb") as handle:
                        handle.write(uploaded_music.getbuffer())
                progress.progress(20, text="Generating natural narration...")
                asyncio.run(_make_tts(script.strip(), voice_name, narration_path))
                progress.progress(45, text="Rendering vertical video and captions...")
                _render_video(
                    source_path, narration_path, output_path, script.strip(),
                    music_path, original_volume, music_volume,
                )
                progress.progress(100, text="Your video is ready.")
                with open(output_path, "rb") as handle:
                    video_bytes = handle.read()
                st.success("Video created successfully.")
                st.video(video_bytes)
                st.download_button(
                    "Download MP4", data=video_bytes, file_name="movie-commentary.mp4",
                    mime="video/mp4", use_container_width=True,
                )
        except Exception as error:
            st.error(f"Video creation failed: {type(error).__name__}: {error}")
        finally:
            progress.empty()
