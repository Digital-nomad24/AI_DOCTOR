import os
import platform
import tempfile
import winsound

from deepgram import DeepgramClient
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from gtts import gTTS
from pydub import AudioSegment

# Load ElevenLabs API key from environment variable
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")


def _server_playback_enabled():
    return os.environ.get("AI_DOCTOR_SERVER_PLAY_AUDIO", "0").lower() in ("1", "true", "yes")


def ensure_folder_exists(filepath):
    """Ensure the directory for the output file exists."""
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def play_audio(output_filepath):
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            os.system(f'afplay "{output_filepath}"')
        elif os_name == "Windows":
            # Convert MP3 to WAV for Windows playback
            sound = AudioSegment.from_file(output_filepath, format="mp3")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                wav_path = temp_wav.name
                sound.export(wav_path, format="wav")
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            raise OSError("Unsupported OS")
    except Exception as e:
        print(f"Error playing audio: {e}")


def text_to_speech_with_gtts(input_text, output_filepath):
    ensure_folder_exists(output_filepath)
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    if _server_playback_enabled():
        play_audio(output_filepath)


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    ensure_folder_exists(output_filepath)
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Laura",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    save(audio, output_filepath)
    if _server_playback_enabled():
        play_audio(output_filepath)


def text_to_speech_with_deepgram(input_text, output_filepath):
    ensure_folder_exists(output_filepath)
    model = (
        os.environ.get("AI_DOCTOR_DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en"
    ).strip() or "aura-2-thalia-en"
    client = DeepgramClient()
    audio_iter = client.speak.v1.audio.generate(text=input_text, model=model)
    body = b"".join(audio_iter)
    if not body:
        raise RuntimeError("Deepgram TTS returned empty audio")
    with open(output_filepath, "wb") as audio_file:
        audio_file.write(body)
    if _server_playback_enabled():
        play_audio(output_filepath)
