import subprocess
import sys
from pathlib import Path

requirements_path = Path(__file__).parent / "requirements.txt"

if requirements_path.exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])

# ChatterBox TTS has strict version pins that conflict with ComfyUI
# Install it without deps, then install the actual required deps separately
try:
    import chatterbox
except ImportError:
    print("[Latent Astronaut] Installing chatterbox-tts (without conflicting deps)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "chatterbox-tts"])
    # Install the deps that chatterbox actually needs (ComfyUI provides torch, torchaudio, etc.)
    # Note: watermarking (resemble-perth) is skipped due to segfault issues with version mismatches
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
