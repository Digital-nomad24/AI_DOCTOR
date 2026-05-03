#Step1: Setup Audio recorder (ffmpeg & portaudio)
# ffmpeg, portaudio, pyaudio
import logging
from io import BytesIO

import speech_recognition as sr
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Simplified function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_lfimit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

audio_filepath="patient_voice_test_for_patient.mp3"
#record_audio(file_path=audio_filepath)

#Step2: Setup Speech to text–STT–model for transcription
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from deepgram import DeepgramClient
from deepgram.types.listen_v1accepted_response import ListenV1AcceptedResponse
from deepgram.types.listen_v1response import ListenV1Response
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
stt_model = "whisper-large-v3"


def _parse_deepgram_listen_response(resp) -> str:
    if isinstance(resp, ListenV1AcceptedResponse):
        raise RuntimeError(
            "Deepgram returned async job id; expected synchronous transcript for file upload."
        )
    if not isinstance(resp, ListenV1Response):
        return ""
    try:
        channels = resp.results.channels
        if not channels:
            return ""
        alts = channels[0].alternatives
        if not alts:
            return ""
        text = alts[0].transcript
        return (text or "").strip()
    except (AttributeError, IndexError):
        return ""


def mono_audio_to_wav_bytes(sr: int, mono: np.ndarray) -> bytes:
    """Write mono float32 [-1, 1] or float64 samples as PCM_16 WAV bytes."""
    mono = np.asarray(mono, dtype=np.float64)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    buf = BytesIO()
    sf.write(buf, pcm, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe_with_deepgram_bytes(audio_bytes: bytes) -> str:
    """Transcribe audio bytes (e.g. WAV) via Deepgram pre-recorded API (uses DEEPGRAM_API_KEY)."""
    model = (os.environ.get("AI_DOCTOR_DEEPGRAM_STT_MODEL") or "nova-3").strip() or "nova-3"
    client = DeepgramClient()
    resp = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model=model,
        language="en",
        smart_format=True,
    )
    return _parse_deepgram_listen_response(resp)


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    client = Groq(api_key=GROQ_API_KEY)

    audio_file = open(audio_filepath, "rb")
    transcription = client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en",
    )

    return transcription.text


def transcribe_with_deepgram(audio_filepath: str) -> str:
    """Transcribe a local file path via Deepgram."""
    return transcribe_with_deepgram_bytes(Path(audio_filepath).read_bytes())