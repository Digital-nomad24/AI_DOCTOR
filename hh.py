"""Entry point with ngrok public URL — same UI as main.py."""

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from demo_app import launch_gradio

    launch_gradio(use_ngrok=True)
