"""
ChatterBox TTS and Voice Conversion nodes for ComfyUI.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import folder_paths

# Force flush prints immediately for debugging crashes
def _log(msg):
    print(f"[ChatterBox] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


# Register chatterbox model folder
CHATTERBOX_DIR = os.path.join(folder_paths.models_dir, "chatterbox")
os.makedirs(CHATTERBOX_DIR, exist_ok=True)


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _list_files(extension: str, subdir: str = None):
    """List files with given extension in chatterbox folder."""
    search_dir = CHATTERBOX_DIR if subdir is None else os.path.join(CHATTERBOX_DIR, subdir)
    files = []

    if os.path.exists(search_dir):
        for root, dirs, filenames in os.walk(search_dir, followlinks=True):
            for f in filenames:
                if f.endswith(extension):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, CHATTERBOX_DIR)
                    files.append(rel_path)

    return sorted(files) if files else ["none"]


def _get_safetensor_files():
    return _list_files(".safetensors")


def _get_pt_files():
    return _list_files(".pt")


def _get_json_files():
    return _list_files(".json")


def _audio_to_file(audio_dict):
    """Convert ComfyUI AUDIO dict to temp wav file path."""
    import soundfile as sf

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]

    wav = waveform[0]  # [C, S]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # [1, S]

    # Convert to numpy [S, C] for soundfile (torchaudio.save is buggy)
    wav_np = wav.cpu().numpy().T

    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, wav_np, sr)
    return f.name


def _tensor_to_audio(wav_tensor, sr):
    """Convert ChatterBox output tensor to ComfyUI AUDIO dict."""
    # Ensure shape is [B, C, S] = [1, 1, S]
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)  # [S] -> [1, 1, S]
    elif wav_tensor.dim() == 2:
        wav_tensor = wav_tensor.unsqueeze(0)  # [C, S] -> [1, C, S]
    return {
        "waveform": wav_tensor,
        "sample_rate": sr
    }


class ChatterBoxTTSLoader:
    """
    Load ChatterBox TTS model for voice cloning text-to-speech.
    Place model files in: ComfyUI/models/chatterbox/
    Required: s3gen.safetensors, t3_cfg.safetensors, ve.safetensors, tokenizer.json
    Optional: conds.pt (built-in voice)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3gen": (_get_safetensor_files(), {
                    "tooltip": "s3gen.safetensors - Speech generation model"
                }),
                "t3_cfg": (_get_safetensor_files(), {
                    "tooltip": "t3_cfg.safetensors - Text-to-token model"
                }),
                "ve": (_get_safetensor_files(), {
                    "tooltip": "ve.safetensors - Voice encoder model"
                }),
                "tokenizer": (_get_json_files(), {
                    "tooltip": "tokenizer.json - Text tokenizer"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
            },
            "optional": {
                "conds": (_get_pt_files(), {
                    "tooltip": "conds.pt - Built-in voice conditionals (optional)"
                }),
            },
        }

    RETURN_TYPES = ("CHATTERBOX_TTS",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio/latent-astronaut"

    def load(self, s3gen: str, t3_cfg: str, ve: str, tokenizer: str, device: str, conds: str = None) -> tuple:
        from safetensors.torch import load_file

        # Lazy imports from chatterbox
        from chatterbox.models.t3 import T3
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.tokenizers import EnTokenizer
        from chatterbox.models.voice_encoder import VoiceEncoder
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.tts import Conditionals

        if device == "auto":
            device = _get_device()

        # Resolve paths
        s3gen_path = os.path.join(CHATTERBOX_DIR, s3gen)
        t3_cfg_path = os.path.join(CHATTERBOX_DIR, t3_cfg)
        ve_path = os.path.join(CHATTERBOX_DIR, ve)
        tokenizer_path = os.path.join(CHATTERBOX_DIR, tokenizer)
        conds_path = os.path.join(CHATTERBOX_DIR, conds) if conds and conds != "none" else None

        # Load voice encoder
        ve_model = VoiceEncoder()
        ve_model.load_state_dict(load_file(ve_path))
        ve_model.to(device).eval()

        # Load T3 model (use strict=False - some t3 checkpoints have different vocab sizes)
        t3 = T3()
        t3_state = load_file(t3_cfg_path)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        missing, unexpected = t3.load_state_dict(t3_state, strict=False)
        if missing:
            _log(f"WARNING: T3 missing keys (may need t3_cfg.safetensors): {missing[:3]}...")
        t3.to(device).eval()

        # Load S3Gen model
        s3gen_model = S3Gen()
        s3gen_model.load_state_dict(load_file(s3gen_path), strict=False)
        s3gen_model.to(device).eval()

        # Load tokenizer
        text_tokenizer = EnTokenizer(tokenizer_path)

        # Load conditionals if provided
        loaded_conds = None
        if conds_path and os.path.exists(conds_path):
            map_location = torch.device('cpu') if device in ["cpu", "mps"] else None
            loaded_conds = Conditionals.load(conds_path, map_location=map_location).to(device)

        # Create a model wrapper object
        class TTSModel:
            def __init__(self):
                self.sr = S3GEN_SR
                self.t3 = t3
                self.s3gen = s3gen_model
                self.ve = ve_model
                self.tokenizer = text_tokenizer
                self.device = device
                self.conds = loaded_conds
                # Skip watermarker - causes segfaults with version mismatches
                self.watermarker = None

        model = TTSModel()
        return (model,)


class ChatterBoxVCLoader:
    """
    Load ChatterBox Voice Conversion model.
    Place model files in: ComfyUI/models/chatterbox/
    Required: s3gen.safetensors
    Optional: conds.pt (built-in voice)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3gen": (_get_safetensor_files(), {
                    "tooltip": "s3gen.safetensors - Speech generation model"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
            },
            "optional": {
                "conds": (_get_pt_files(), {
                    "tooltip": "conds.pt - Built-in voice conditionals (optional)"
                }),
            },
        }

    RETURN_TYPES = ("CHATTERBOX_VC",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio/latent-astronaut"

    def load(self, s3gen: str, device: str, conds: str = None) -> tuple:
        from safetensors.torch import load_file

        from chatterbox.models.s3gen import S3Gen, S3GEN_SR

        if device == "auto":
            device = _get_device()

        s3gen_path = os.path.join(CHATTERBOX_DIR, s3gen)
        conds_path = os.path.join(CHATTERBOX_DIR, conds) if conds and conds != "none" else None

        # Load S3Gen model
        s3gen_model = S3Gen()
        s3gen_model.load_state_dict(load_file(s3gen_path), strict=False)
        s3gen_model.to(device).eval()

        # Load conditionals if provided
        ref_dict = None
        if conds_path and os.path.exists(conds_path):
            map_location = torch.device('cpu') if device in ["cpu", "mps"] else None
            states = torch.load(conds_path, map_location=map_location)
            ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in states['gen'].items()
            }

        # Create a model wrapper object
        class VCModel:
            def __init__(self):
                self.sr = S3GEN_SR
                self.s3gen = s3gen_model
                self.device = device
                self.ref_dict = ref_dict
                # Skip watermarker - causes segfaults with version mismatches
                self.watermarker = None

        model = VCModel()
        return (model,)


class ChatterBoxTTS:
    """
    Text-to-Speech using ChatterBox.
    Clones a voice from reference audio and generates speech from text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CHATTERBOX_TTS", {
                    "tooltip": "ChatterBox TTS model from loader"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of voice cloning.",
                    "tooltip": "Text to synthesize into speech"
                }),
                "voice_audio": ("AUDIO", {
                    "tooltip": "Reference audio for voice cloning (6-10 seconds ideal)"
                }),
                "exaggeration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Emotion/expression intensity (0=neutral, 1=exaggerated)"
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Classifier-free guidance weight (higher=more adherence to text)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (higher=more varied/random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold for sampling"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeating tokens (1.0=no penalty)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0=random)"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/latent-astronaut"

    def generate(
        self,
        model,
        text: str,
        voice_audio: dict,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> tuple:
        _log("TTS generate() called")
        _log(f"  text length: {len(text)}, seed: {seed}")
        _log(f"  params: exag={exaggeration}, cfg={cfg_weight}, temp={temperature}")

        try:
            import torch.nn.functional as F
            _log("  imported torch.nn.functional")

            import librosa
            _log("  imported librosa")

            from chatterbox.models.s3tokenizer import S3_SR
            _log("  imported S3_SR")

            from chatterbox.models.s3gen import S3GEN_SR
            _log("  imported S3GEN_SR")

            from chatterbox.models.t3.modules.cond_enc import T3Cond
            _log("  imported T3Cond")

            from chatterbox.models.s3tokenizer import drop_invalid_tokens
            _log("  imported drop_invalid_tokens")

            from chatterbox.tts import punc_norm
            _log("  imported punc_norm")
        except Exception as e:
            _log(f"  IMPORT ERROR: {e}")
            _log(traceback.format_exc())
            raise

        ENC_COND_LEN = 6 * S3_SR
        DEC_COND_LEN = 10 * S3GEN_SR
        _log(f"  ENC_COND_LEN={ENC_COND_LEN}, DEC_COND_LEN={DEC_COND_LEN}")

        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            _log(f"  seeded RNG with {seed}")

        _log("  converting voice_audio to file...")
        voice_path = _audio_to_file(voice_audio)
        _log(f"  voice_path: {voice_path}")

        try:
            _log("  loading reference audio with librosa...")
            s3gen_ref_wav, _ = librosa.load(voice_path, sr=S3GEN_SR)
            _log(f"  s3gen_ref_wav shape: {s3gen_ref_wav.shape}")

            _log("  resampling to 16k...")
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
            _log(f"  ref_16k_wav shape: {ref_16k_wav.shape}")

            s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
            _log(f"  trimmed s3gen_ref_wav shape: {s3gen_ref_wav.shape}")

            _log("  calling s3gen.embed_ref...")
            s3gen_ref_dict = model.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=model.device)
            _log(f"  s3gen_ref_dict keys: {s3gen_ref_dict.keys() if hasattr(s3gen_ref_dict, 'keys') else type(s3gen_ref_dict)}")

            # Speech cond prompt tokens
            _log("  getting speech cond prompt tokens...")
            if plen := model.t3.hp.speech_cond_prompt_len:
                _log(f"  plen={plen}")
                s3_tokzr = model.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(model.device)
                _log(f"  t3_cond_prompt_tokens shape: {t3_cond_prompt_tokens.shape}")

            # Voice-encoder speaker embedding
            _log("  getting voice encoder embedding...")
            ve_embed = torch.from_numpy(model.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(model.device)
            _log(f"  ve_embed shape: {ve_embed.shape}")

            _log("  creating T3Cond...")
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)
            _log("  T3Cond created")

            # Norm and tokenize text
            _log("  normalizing and tokenizing text...")
            text = punc_norm(text)
            text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)
            _log(f"  text_tokens shape: {text_tokens.shape}")

            if cfg_weight > 0.0:
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                _log(f"  doubled text_tokens for CFG, shape: {text_tokens.shape}")

            sot = model.t3.hp.start_text_token
            eot = model.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            _log(f"  padded text_tokens shape: {text_tokens.shape}")

            _log("  starting inference_mode...")
            with torch.inference_mode():
                _log("  calling t3.inference...")
                speech_tokens = model.t3.inference(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
                _log(f"  t3.inference returned, type: {type(speech_tokens)}")

                speech_tokens = speech_tokens[0]
                _log(f"  speech_tokens[0] shape: {speech_tokens.shape}")

                speech_tokens = drop_invalid_tokens(speech_tokens)
                _log(f"  after drop_invalid, shape: {speech_tokens.shape}")

                speech_tokens = speech_tokens[speech_tokens < 6561]
                _log(f"  after filtering, shape: {speech_tokens.shape}")

                speech_tokens = speech_tokens.to(model.device)
                _log("  speech_tokens moved to device")

                _log("  calling s3gen.inference...")
                wav, _ = model.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref_dict,
                )
                _log(f"  s3gen.inference returned wav shape: {wav.shape}")

                wav = wav.squeeze(0).detach().cpu().numpy()
                _log(f"  wav numpy shape: {wav.shape}")

                # Skip watermark - causes segfaults
                final_wav = wav

            _log("  converting to audio output...")
            audio_out = _tensor_to_audio(torch.from_numpy(final_wav), model.sr)
            _log("  audio_out created successfully")
        except Exception as e:
            _log(f"GENERATE ERROR: {e}")
            _log(traceback.format_exc())
            raise
        finally:
            _log(f"  cleaning up temp file {voice_path}")
            os.unlink(voice_path)

        _log("TTS generate() complete!")
        return (audio_out,)


class ChatterBoxVC:
    """
    Voice Conversion using ChatterBox.
    Converts audio to sound like a target voice.
    Note: VC has no sampling parameters - it's a direct conversion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CHATTERBOX_VC", {
                    "tooltip": "ChatterBox VC model from loader"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio to convert"
                }),
                "target_voice": ("AUDIO", {
                    "tooltip": "Target voice reference audio (6-10 seconds ideal)"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = "audio/latent-astronaut"

    def convert(
        self,
        model,
        audio: dict,
        target_voice: dict,
    ) -> tuple:
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.s3gen import S3GEN_SR

        DEC_COND_LEN = 10 * S3GEN_SR

        audio_path = _audio_to_file(audio)
        voice_path = _audio_to_file(target_voice)

        try:
            # Load target voice and create ref_dict
            s3gen_ref_wav, _ = librosa.load(voice_path, sr=S3GEN_SR)
            s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
            ref_dict = model.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=model.device)

            with torch.inference_mode():
                audio_16, _ = librosa.load(audio_path, sr=S3_SR)
                audio_16 = torch.from_numpy(audio_16).float().to(model.device)[None, ]

                s3_tokens, _ = model.s3gen.tokenizer(audio_16)
                wav, _ = model.s3gen.inference(
                    speech_tokens=s3_tokens,
                    ref_dict=ref_dict,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                # Skip watermark - causes segfaults
                final_wav = wav

            audio_out = _tensor_to_audio(torch.from_numpy(final_wav), model.sr)
        finally:
            os.unlink(audio_path)
            os.unlink(voice_path)

        return (audio_out,)


# =============================================================================
# SIMPLE AUTO-DOWNLOAD LOADERS (use official from_pretrained API)
# =============================================================================

class ChatterBoxTTSLoaderAuto:
    """
    Auto-download ChatterBox TTS model from HuggingFace.
    Downloads on first use, cached for subsequent runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
            },
        }

    RETURN_TYPES = ("CHATTERBOX_TTS",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio/latent-astronaut"

    def load(self, device: str) -> tuple:
        from chatterbox.tts import ChatterboxTTS

        if device == "auto":
            device = _get_device()

        _log(f"Loading ChatterBox TTS (auto-download) on {device}...")
        model = ChatterboxTTS.from_pretrained(device=device)
        # Disable watermarker (causes segfaults)
        model.watermarker = None
        _log("ChatterBox TTS loaded successfully")

        return (model,)


class ChatterBoxVCLoaderAuto:
    """
    Auto-download ChatterBox VC model from HuggingFace.
    Downloads on first use, cached for subsequent runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
            },
        }

    RETURN_TYPES = ("CHATTERBOX_VC",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio/latent-astronaut"

    def load(self, device: str) -> tuple:
        from chatterbox.vc import ChatterboxVC

        if device == "auto":
            device = _get_device()

        _log(f"Loading ChatterBox VC (auto-download) on {device}...")
        model = ChatterboxVC.from_pretrained(device=device)
        # Disable watermarker (causes segfaults)
        model.watermarker = None
        _log("ChatterBox VC loaded successfully")

        return (model,)


class ChatterBoxTTSSimple:
    """
    Simple TTS using auto-loaded ChatterBox model.
    Uses the official generate() API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CHATTERBOX_TTS", {
                    "tooltip": "ChatterBox TTS model from auto-loader"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of voice cloning.",
                    "tooltip": "Text to synthesize into speech"
                }),
                "voice_audio": ("AUDIO", {
                    "tooltip": "Reference audio for voice cloning (6-10 seconds ideal)"
                }),
                "exaggeration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Emotion/expression intensity"
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Classifier-free guidance weight"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Sampling temperature"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/latent-astronaut"

    def generate(self, model, text: str, voice_audio: dict,
                 exaggeration: float, cfg_weight: float, temperature: float) -> tuple:
        _log("TTS Simple generate() called")

        voice_path = _audio_to_file(voice_audio)

        try:
            _log(f"  Generating speech for: {text[:50]}...")
            wav = model.generate(
                text=text,
                audio_prompt_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            _log(f"  Generated wav shape: {wav.shape}")

            # wav is tensor [1, samples], convert to numpy
            wav_np = wav.squeeze(0).cpu().numpy()
            audio_out = _tensor_to_audio(torch.from_numpy(wav_np), model.sr)
            _log("  Done!")
        finally:
            os.unlink(voice_path)

        return (audio_out,)


class ChatterBoxVCSimple:
    """
    Simple Voice Conversion using auto-loaded ChatterBox model.
    Uses the official generate() API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CHATTERBOX_VC", {
                    "tooltip": "ChatterBox VC model from auto-loader"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio to convert"
                }),
                "target_voice": ("AUDIO", {
                    "tooltip": "Target voice reference audio"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = "audio/latent-astronaut"

    def convert(self, model, audio: dict, target_voice: dict) -> tuple:
        _log("VC Simple convert() called")

        audio_path = _audio_to_file(audio)
        voice_path = _audio_to_file(target_voice)

        try:
            _log("  Converting voice...")
            wav = model.generate(
                audio=audio_path,
                target_voice_path=voice_path,
            )
            _log(f"  Generated wav shape: {wav.shape}")

            wav_np = wav.squeeze(0).cpu().numpy()
            audio_out = _tensor_to_audio(torch.from_numpy(wav_np), model.sr)
            _log("  Done!")
        finally:
            os.unlink(audio_path)
            os.unlink(voice_path)

        return (audio_out,)


NODE_CLASS_MAPPINGS = {
    # Manual loaders (pick individual files)
    "ChatterBoxTTSLoader": ChatterBoxTTSLoader,
    "ChatterBoxVCLoader": ChatterBoxVCLoader,
    "ChatterBoxTTS": ChatterBoxTTS,
    "ChatterBoxVC": ChatterBoxVC,
    # Auto loaders (download from HuggingFace)
    "ChatterBoxTTSLoaderAuto": ChatterBoxTTSLoaderAuto,
    "ChatterBoxVCLoaderAuto": ChatterBoxVCLoaderAuto,
    "ChatterBoxTTSSimple": ChatterBoxTTSSimple,
    "ChatterBoxVCSimple": ChatterBoxVCSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Manual loaders
    "ChatterBoxTTSLoader": "ChatterBox TTS Loader (Manual)",
    "ChatterBoxVCLoader": "ChatterBox VC Loader (Manual)",
    "ChatterBoxTTS": "ChatterBox TTS (Manual)",
    "ChatterBoxVC": "ChatterBox VC (Manual)",
    # Auto loaders
    "ChatterBoxTTSLoaderAuto": "ChatterBox TTS Loader (Auto)",
    "ChatterBoxVCLoaderAuto": "ChatterBox VC Loader (Auto)",
    "ChatterBoxTTSSimple": "ChatterBox TTS (Simple)",
    "ChatterBoxVCSimple": "ChatterBox VC (Simple)",
}
