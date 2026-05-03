"""Primary entry point: runs the Gradio demo on port 7865.

Public tunnel: set environment variable USE_NGROK=1 (or run `python hh.py`).
"""
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from demo_app import launch_gradio

    launch_gradio()
