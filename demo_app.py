"""Shared Gradio UI and diagnosis pipeline for the AI Doctor demo."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from brain_of_the_doctor import analyze_image_with_query, encode_image
from breast_cancer_classifer import breast_cancer_detection_model
from voice_of_the_doctor import (
    text_to_speech_with_deepgram,
    text_to_speech_with_elevenlabs,
    text_to_speech_with_gtts,
)
from voice_of_the_patient import transcribe_with_deepgram

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_ID = (os.environ.get("AI_DOCTOR_GROQ_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

_NO_INPUT_HINT = "Please provide at least an image, a short description, or record voice to capture a transcript before analyzing."

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def contains_hsi_keywords(speech_text: str) -> bool:
    keywords = ["hsi", "hyperspectral", "hyperspectral imaging", "tissue"]
    lower = speech_text.lower()
    return any(k in lower for k in keywords)


def is_hsi_image(image_path: str | None) -> bool:
    return bool(image_path) and image_path.lower().endswith((".mat", ".npy", ".hdr"))


def _tts_backend():
    raw = os.environ.get("AI_DOCTOR_TTS", "gtts").strip().lower()
    if raw in {"elevenlabs", "gtts", "deepgram"}:
        return raw
    return "gtts"


def _tts_enabled() -> bool:
    raw = os.environ.get("AI_DOCTOR_ENABLE_TTS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _synthesize_voice(diagnosis: str, audio_output_path: str) -> None:
    backend = _tts_backend()
    if backend == "gtts":
        text_to_speech_with_gtts(diagnosis, audio_output_path)
    elif backend == "elevenlabs":
        text_to_speech_with_elevenlabs(diagnosis, audio_output_path)
    else:
        text_to_speech_with_deepgram(diagnosis, audio_output_path)


def transcribe_recorded_audio(audio_path, current_transcript):
    """One-shot mic transcription after recording stops."""
    current_transcript = (current_transcript or "").strip()
    if not os.environ.get("DEEPGRAM_API_KEY"):
        return "[STT unavailable: set DEEPGRAM_API_KEY in .env]"

    if not audio_path:
        return current_transcript

    try:
        text = transcribe_with_deepgram(audio_path)
        return (text or current_transcript).strip()
    except Exception:
        logger.exception("Recorded transcription failed")
        return current_transcript


def process_inputs(live_transcript, description, image_pil, progress: gr.Progress = gr.Progress()):
    diagnosis = ""
    route_mode = "—"
    audio_output_path: str | None = None

    try:
        logger.info("Consult: handler started (analyze click received).")
        Path("static").mkdir(parents=True, exist_ok=True)

        progress(0.05, desc="Preparing…")

        speech_text = (live_transcript or "").strip()
        desc = (description or "").strip()
        context_text = " ".join(p for p in (speech_text, desc) if p)

        if not diagnosis:
            image_filepath = None
            encoded_image = None
            if image_pil is not None:
                progress(0.3, desc="Saving image…")
                image_filepath = "static/temp_image.png"
                image_pil.save(image_filepath)
                encoded_image = encode_image(image_filepath)

            wants_classifier = contains_hsi_keywords(context_text) or (
                image_filepath and is_hsi_image(image_filepath)
            )

            if wants_classifier:
                if not image_filepath:
                    route_mode = "Tissue classifier (demo) — image required"
                    diagnosis = (
                        "For the tissue specialist model in this demo, please upload an image "
                        "(PNG or JPG). Raw hyperspectral formats like .mat, .npy, or .hdr can be "
                        "used when provided as files the app can load."
                    )
                else:
                    progress(0.45, desc="Running tissue classifier…")
                    route_mode = "Tissue classifier (demo)"
                    diagnosis = breast_cancer_detection_model(image_filepath)

            elif encoded_image and context_text:
                progress(0.5, desc="Vision + language model…")
                route_mode = "Vision + language model (Groq)"
                diagnosis = analyze_image_with_query(
                    query=system_prompt + " " + context_text,
                    encoded_image=encoded_image,
                    model=MODEL_ID,
                    image_media_type="image/png",
                )
            elif encoded_image:
                progress(0.5, desc="Vision + language model…")
                route_mode = "Vision + language model (Groq)"
                diagnosis = analyze_image_with_query(
                    query=system_prompt,
                    encoded_image=encoded_image,
                    model=MODEL_ID,
                    image_media_type="image/png",
                )
            elif context_text:
                progress(0.5, desc="Language model…")
                route_mode = "Language model — text only (Groq)"
                diagnosis = analyze_image_with_query(
                    query=system_prompt + " " + context_text,
                    encoded_image=None,
                    model=MODEL_ID,
                )
            else:
                route_mode = "—"
                diagnosis = _NO_INPUT_HINT

        logger.info("Consult: pipeline done route=%r response_chars=%s", route_mode, len(diagnosis) if diagnosis else 0)

        if _tts_enabled() and diagnosis and diagnosis != _NO_INPUT_HINT and route_mode != "Error":
            progress(0.75, desc="Generating voice…")
            audio_output_path = str(Path("static") / f"tts_{uuid.uuid4().hex}.mp3")
            try:
                _synthesize_voice(diagnosis, audio_output_path)
            except Exception as tts_err:
                logger.exception("TTS failed (response text unchanged).")
                route_mode = f"{route_mode} · TTS failed: {type(tts_err).__name__}"

        progress(1.0, desc="Done")

    except Exception as e:
        logger.exception("Consult: pipeline error.")
        diagnosis = f"{type(e).__name__}: {e}"
        route_mode = "Error"

    audio_path = audio_output_path if (audio_output_path and os.path.exists(audio_output_path)) else None
    download_val = audio_path if audio_path else None

    return diagnosis, audio_path, download_val, route_mode


def clear_all():
    return (
        None,
        gr.update(value=""),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value="—"),
    )


# ─── Premium Design System ────────────────────────────────────────────────────
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --c-bg:          #080d14;
    --c-surface:     #0d1520;
    --c-surface2:    #111d2e;
    --c-border:      rgba(99, 179, 237, 0.10);
    --c-border-hi:   rgba(99, 179, 237, 0.28);
    --c-teal:        #2dd4c8;
    --c-teal-dim:    #1a9e96;
    --c-blue:        #60a5fa;
    --c-text:        #e2eaf6;
    --c-muted:       #7a93b8;
    --c-danger:      #fbbf24;
    --c-glow:        rgba(45, 212, 200, 0.18);
    --c-glow-blue:   rgba(96, 165, 250, 0.12);
    --r-card:        18px;
    --r-btn:         12px;
    --shadow-card:   0 8px 40px rgba(0,0,0,0.55), 0 1px 0 rgba(99,179,237,0.06) inset;
    --shadow-glow:   0 0 40px var(--c-glow);
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --transition:    0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ── */
html, body {
    margin: 0 !important;
    width: 100% !important;
    min-height: 100% !important;
}

body, .gradio-container {
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: var(--font-body) !important;
}

/* Animated star-field background */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(45,212,200,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(96,165,250,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(13,21,32,0.9) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 1.5rem clamp(1rem, 4vw, 2.5rem) 3rem !important;
    position: relative;
    z-index: 1;
}

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 2.8rem 1rem 2rem;
    position: relative;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(45,212,200,0.10);
    border: 1px solid rgba(45,212,200,0.25);
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--c-teal);
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

.hero-eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--c-teal);
    box-shadow: 0 0 8px var(--c-teal);
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}

.gradio-container .hero-wrap h1,
.hero-wrap h1 {
    font-family: var(--font-display) !important;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin: 0 0 0.9rem;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: none !important;
    background-clip: border-box !important;
    -webkit-background-clip: border-box !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
}

.hero-sub {
    color: var(--c-muted);
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto;
}

/* ── Disclaimer ── */
.disclaimer-box {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: linear-gradient(135deg, rgba(251,191,36,0.07) 0%, rgba(251,191,36,0.03) 100%);
    border: 1px solid rgba(251,191,36,0.22);
    border-left: 3px solid #fbbf24;
    padding: 14px 18px;
    border-radius: 12px;
    margin-bottom: 1.6rem;
    font-size: 0.875rem;
    line-height: 1.6;
    color: #fde68a;
    backdrop-filter: blur(6px);
}

.disclaimer-box::before {
    content: '⚠';
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 1px;
}

.disclaimer-box strong { color: #fcd34d; }

/* ── Cards & Panels ── */
.panel-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--r-card);
    box-shadow: var(--shadow-card);
    overflow: hidden;
    transition: border-color var(--transition), box-shadow var(--transition);
}

.panel-card:hover {
    border-color: var(--c-border-hi);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}

/* ── Section labels ── */
.section-label {
    font-family: var(--font-display);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--c-teal);
    margin: 0 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--c-border-hi), transparent);
}

/* ── Route badge ── */
.route-badge-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.4rem;
}

.route-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-muted);
    white-space: nowrap;
}

/* ── Gradio component overrides ── */

/* Labels */
.gradio-container label span,
.gradio-container .label-wrap span {
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--c-muted) !important;
}

/* Textareas & inputs */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background: rgba(13, 21, 32, 0.8) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 12px !important;
    color: var(--c-text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    padding: 12px 14px !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
    resize: none !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: var(--c-teal-dim) !important;
    box-shadow: 0 0 0 3px rgba(45,212,200,0.12), 0 0 20px rgba(45,212,200,0.08) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: rgba(122,147,184,0.45) !important;
}

/* Block containers */
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Wrappers */
.gradio-container .wrap {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-card) !important;
    transition: border-color var(--transition) !important;
    overflow: hidden;
}

.gradio-container .wrap:hover {
    border-color: var(--c-border-hi) !important;
}

/* Image upload area */
.gradio-container .upload-container,
.gradio-container [data-testid="image"] .wrap {
    background: var(--c-surface2) !important;
    border: 1px dashed var(--c-border-hi) !important;
    border-radius: var(--r-card) !important;
    min-height: 200px;
    transition: all var(--transition) !important;
}

.gradio-container .upload-container:hover,
.gradio-container [data-testid="image"] .wrap:hover {
    border-color: var(--c-teal-dim) !important;
    background: rgba(45,212,200,0.03) !important;
}

/* Audio component */
.gradio-container .audio-container,
.gradio-container [data-testid="audio"] .wrap {
    background: var(--c-surface2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-card) !important;
}

/* Buttons — primary */
.gradio-container .gr-button-primary,
.gradio-container button[variant="primary"],
.gradio-container button.primary {
    background: linear-gradient(135deg, #1a9e96 0%, #0f766e 60%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: var(--r-btn) !important;
    color: #fff !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 13px 28px !important;
    position: relative;
    overflow: hidden;
    transition: transform var(--transition), box-shadow var(--transition) !important;
    box-shadow: 0 4px 24px rgba(15,118,110,0.45) !important;
    cursor: pointer !important;
}

.gradio-container .gr-button-primary::before,
.gradio-container button[variant="primary"]::before,
.gradio-container button.primary::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 60%);
    pointer-events: none;
}

.gradio-container .gr-button-primary:hover,
.gradio-container button[variant="primary"]:hover,
.gradio-container button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(15,118,110,0.6) !important;
}

.gradio-container .gr-button-primary:active,
.gradio-container button[variant="primary"]:active,
.gradio-container button.primary:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Buttons — secondary */
.gradio-container .gr-button-secondary,
.gradio-container button[variant="secondary"],
.gradio-container button.secondary {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--c-border-hi) !important;
    border-radius: var(--r-btn) !important;
    color: var(--c-muted) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 13px 24px !important;
    transition: all var(--transition) !important;
    cursor: pointer !important;
}

.gradio-container .gr-button-secondary:hover,
.gradio-container button[variant="secondary"]:hover,
.gradio-container button.secondary:hover {
    background: rgba(255,255,255,0.08) !important;
    color: var(--c-text) !important;
    border-color: rgba(99,179,237,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.gradio-container [role="tablist"] {
    background: rgba(13,21,32,0.6) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
    display: flex !important;
    margin-bottom: 1.5rem !important;
    backdrop-filter: blur(8px) !important;
}

.gradio-container [role="tab"] {
    border-radius: 10px !important;
    padding: 9px 20px !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--c-muted) !important;
    border: none !important;
    background: transparent !important;
    transition: all var(--transition) !important;
    letter-spacing: 0.03em;
}

.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="tab"].selected {
    background: linear-gradient(135deg, rgba(45,212,200,0.15), rgba(96,165,250,0.10)) !important;
    color: var(--c-teal) !important;
    box-shadow: 0 2px 12px rgba(45,212,200,0.12) !important;
}

/* Accordion */
.gradio-container .accordion {
    background: var(--c-surface2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
}

.gradio-container .accordion > .label-wrap {
    border-bottom: 1px solid var(--c-border) !important;
    padding: 14px 16px !important;
}

/* File download component */
.gradio-container .file-preview {
    background: var(--c-surface2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 12px !important;
}

/* Row spacing */
.gradio-container .row {
    gap: 1.1rem !important;
}

/* Markdown inside blocks */
.gradio-container .md p,
.gradio-container .md li,
.gradio-container .md h3 {
    color: var(--c-text) !important;
    font-family: var(--font-body) !important;
}

.gradio-container .md h3 {
    font-family: var(--font-display) !important;
    color: var(--c-teal) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
}

/* Progress bar */
.gradio-container .progress-bar {
    background: linear-gradient(90deg, var(--c-teal), var(--c-blue)) !important;
    border-radius: 100px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--c-border-hi); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-teal-dim); }

/* ── Header divider ── */
.header-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, var(--c-teal), var(--c-blue));
    border-radius: 100px;
    margin: 1rem auto 0;
    box-shadow: 0 0 16px var(--c-glow);
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1.4rem 0 0.4rem;
    flex-wrap: wrap;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--c-teal);
    line-height: 1;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-muted);
}

/* ── Column headers ── */
.col-header {
    font-family: var(--font-display);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--c-teal);
    padding: 0 2px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.col-header .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--c-teal);
    box-shadow: 0 0 6px var(--c-teal);
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .gradio-container { padding: 1rem 0.75rem 2rem !important; }
    .stats-strip { gap: 1.5rem; }
    .hero-wrap { padding: 1.8rem 0.5rem 1.4rem; }
}
"""


def _app_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        spacing_size="md",
        radius_size="lg",
        text_size="md",
        font=[
            gr.themes.GoogleFont("DM Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    ).set(
        # Override Base theme tokens to match dark palette
        body_background_fill="#080d14",
        body_text_color="#e2eaf6",
        block_background_fill="#0d1520",
        block_border_color="rgba(99,179,237,0.10)",
        block_label_text_color="#7a93b8",
        input_background_fill="#0d1520",
        input_border_color="rgba(99,179,237,0.10)",
        button_primary_background_fill="linear-gradient(135deg, #1a9e96, #0f766e)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.04)",
        button_secondary_border_color="rgba(99,179,237,0.28)",
        button_secondary_text_color="#7a93b8",
    )


def create_interface():
    with gr.Blocks(
        theme=_app_theme(),
        css=APP_CSS,
        fill_width=True,
        fill_height=True,
        title="AI Doctor",
    ) as iface:

        gr.HTML("""
<div class="hero-wrap">
  <h1>AI Doctor</h1>
</div>
""")

        # ── Disclaimer ───────────────────────────────────────────────────────
        gr.HTML("""
<div class="disclaimer-box">
  <div><strong>Educational prototype only.</strong> This is not medical advice, not FDA-cleared software, and not a substitute for a licensed clinician or emergency care. Use for demonstrations and learning only.</div>
</div>
""")

        with gr.Row():
            with gr.Column(scale=1):
                mode_badge = gr.Textbox(
                    label="Routing mode",
                    value="—",
                    interactive=False,
                    max_lines=1,
                    show_label=False,
                    placeholder="Awaiting input…",
                )

        gr.HTML("<div style='height:4px'></div>")

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=320):
                image_input = gr.Image(
                    type="pil",
                    label="Medical image (PNG / JPG)",
                    show_label=True,
                    height=230,
                )

                gr.HTML("<div style='height:10px'></div>")

                description_input = gr.Textbox(
                    label="Describe symptoms or what to look for",
                    lines=4,
                    placeholder="Optional: add context for the image (e.g. pain, duration, what you want checked).",
                    show_label=True,
                )

                gr.HTML("<div style='height:10px'></div>")

                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    format="wav",
                    recording=False,
                    label="Microphone — click Record, then Stop to transcribe",
                    show_label=True,
                )

                gr.HTML("<div style='height:14px'></div>")

                with gr.Row():
                    analyze_btn = gr.Button(
                        "⚡  Analyze & Speak",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "✕  Clear",
                        variant="secondary",
                        size="lg",
                    )

            with gr.Column(scale=1, min_width=320):
                transcript_output = gr.Textbox(
                    label="Voice transcript",
                    lines=4,
                    interactive=True,
                    placeholder="Record with the mic and stop when done; transcript appears here (editable).",
                )

                diagnosis_output = gr.Textbox(
                    label="Doctor's response",
                    lines=6,
                    interactive=False,
                    placeholder="AI assessment will appear here…",
                )

                with gr.Row():
                    audio_output = gr.Audio(
                        label="Spoken response",
                        interactive=False,
                    )

                download_file = gr.File(
                    label="⬇  Download MP3",
                    interactive=False,
                )

        audio_input.stop_recording(
            transcribe_recorded_audio,
            inputs=[audio_input, transcript_output],
            outputs=[transcript_output],
        )

        gr.HTML("<div style='height:8px'></div>")
        with gr.Accordion("ℹ  How this demo works", open=False):
            gr.Markdown("""
### Pipeline Overview

**Step 1 — Image (optional)**  
Upload a PNG or JPG scan or photo. Pixels are encoded and sent to the vision-language model.

**Step 2 — Description (optional)**  
Type a short note with symptoms, context, or what you want assessed.

**Step 3 — Voice transcript (optional)**  
Record with the microphone and stop when done. The app sends one audio file to **Deepgram** and fills the transcript box. Requires `DEEPGRAM_API_KEY`. You can edit the transcript before **Analyze**.

**Step 4 — Routing**  
Certain keywords (*hyperspectral, tissue, hsi*) route the request to a local Keras tissue classifier instead of the LLM.

**Step 5 — Spoken response**  
The doctor reply is turned into MP3 using your `AI_DOCTOR_TTS` setting (`gtts`, `elevenlabs`, or `deepgram`; Deepgram TTS uses `DEEPGRAM_API_KEY` and `AI_DOCTOR_DEEPGRAM_TTS_MODEL` if set). Vision and language models still use Groq where configured (`GROQ_API_KEY`).
""")

        # ── Wire up events ────────────────────────────────────────────────────
        analyze_btn.click(
            fn=process_inputs,
            inputs=[transcript_output, description_input, image_input],
            outputs=[diagnosis_output, audio_output, download_file, mode_badge],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                image_input,
                description_input,
                audio_input,
                transcript_output,
                diagnosis_output,
                audio_output,
                download_file,
                mode_badge,
            ],
        )

    return iface


def launch_gradio(*, use_ngrok: bool | None = None, server_port: int = 7865):
    if use_ngrok is None:
        use_ngrok = os.getenv("USE_NGROK", "0").strip().lower() in ("1", "true", "yes")

    iface = create_interface()

    if use_ngrok:
        from pyngrok import ngrok
        public_url = ngrok.connect(server_port)
        print("Public ngrok URL:", public_url)

    iface.launch(
        debug=True,
        share=False,
        server_port=server_port,
        server_name="0.0.0.0",
        show_error=True,
        prevent_thread_lock=True,
        pwa=True,
    )