"""
Text-to-Speech Module – Multiple Engine Support
================================================
Engines available:
  1) piper      – Neural TTS via subprocess, 100% offline (RECOMMENDED)
  2) pyttsx3    – Windows SAPI voices, 100% offline, instant
  3) kokoro     – Neural TTS, natural voices, 100% offline
  4) edge-tts   – Microsoft Edge voices, very natural, needs internet

Default: piper (fast, offline, natural)
"""

import os
import io
import time
import wave
import subprocess
import asyncio
import threading
import queue
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import TTS_ENGINE, TTS_LANGUAGE

logger = logging.getLogger(__name__)


# ======================================================================
# Engine 1: Piper  (Offline – Neural TTS via subprocess)
# ======================================================================
class PiperEngine:
    """Piper neural TTS via subprocess. Fast, natural, fully offline."""

    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), "piper_models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Check if piper is available
        self.piper_cmd = self._find_piper()
        self.model_path = None
        self.config_path = None
        self._find_model()

        if self.piper_cmd and self.model_path:
            print(f"✅  Piper TTS ready  (model: {os.path.basename(self.model_path)})")
            logger.info("Piper TTS ready: %s", self.model_path)
        elif self.model_path:
            # Model exists but piper not found — try Python piper-tts
            print(f"✅  Piper TTS ready (Python mode)  (model: {os.path.basename(self.model_path)})")
        else:
            print("⚠️  No Piper model found. Downloading a female English voice...")
            self._download_default_model()

    def _find_piper(self) -> str | None:
        """Check if piper CLI is available."""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("Piper CLI found")
                return "piper"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for piper in common locations
        piper_paths = [
            os.path.join(os.path.dirname(__file__), "piper", "piper.exe"),
            os.path.join(os.path.dirname(__file__), "piper.exe"),
        ]
        for p in piper_paths:
            if os.path.exists(p):
                logger.info("Piper found at: %s", p)
                return p

        return None

    def _find_model(self):
        """Look for .onnx model in piper_models/"""
        if not os.path.exists(self.model_dir):
            return
        for f in os.listdir(self.model_dir):
            if f.endswith(".onnx"):
                self.model_path = os.path.join(self.model_dir, f)
                cfg = self.model_path + ".json"
                if os.path.exists(cfg):
                    self.config_path = cfg
                break

    def _download_default_model(self):
        """Download a female English Piper voice model."""
        import urllib.request

        # en_US-amy-medium is a nice female voice
        model_name = "en_US-amy-medium"
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/"

        model_file = f"{model_name}.onnx"
        config_file = f"{model_name}.onnx.json"

        model_path = os.path.join(self.model_dir, model_file)
        config_path = os.path.join(self.model_dir, config_file)

        try:
            print(f"📥  Downloading {model_file}... (one-time, ~60MB)")
            urllib.request.urlretrieve(base_url + model_file, model_path)
            print(f"📥  Downloading {config_file}...")
            urllib.request.urlretrieve(base_url + config_file, config_path)
            self.model_path = model_path
            self.config_path = config_path
            print(f"✅  Piper model downloaded!")
            logger.info("Piper model downloaded: %s", model_name)
        except Exception as e:
            print(f"❌  Download failed: {e}")
            print("    You can manually download a model from:")
            print("    https://huggingface.co/rhasspy/piper-voices")
            print(f"    Place .onnx + .onnx.json files in: {self.model_dir}")
            logger.error("Piper model download failed: %s", e)

    def speak(self, text: str, output_file: str = "output_speech.wav"):
        if not text.strip() or not self.model_path:
            return

        t0 = time.time()

        # Try subprocess method first (faster)
        if self.piper_cmd:
            try:
                self._speak_subprocess(text, output_file)
                elapsed = time.time() - t0
                print(f"🔊  TTS generated in {elapsed:.1f}s")
                logger.info("Piper TTS (subprocess): %.1fs", elapsed)
                _play_wav(output_file)
                return
            except Exception as e:
                logger.warning("Piper subprocess failed, trying Python: %s", e)

        # Fallback: Python piper-tts library
        try:
            self._speak_python(text, output_file)
            elapsed = time.time() - t0
            print(f"🔊  TTS generated in {elapsed:.1f}s")
            logger.info("Piper TTS (Python): %.1fs", elapsed)
            _play_wav(output_file)
        except Exception as e:
            print(f"❌  Piper TTS error: {e}")
            logger.error("Piper TTS error: %s", e)

    def _speak_subprocess(self, text: str, output_file: str):
        """Generate speech using Piper CLI subprocess."""
        cmd = [
            self.piper_cmd,
            "--model", self.model_path,
            "--output_file", output_file,
        ]
        if self.config_path:
            cmd.extend(["--config", self.config_path])

        proc = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Piper subprocess error: {proc.stderr}")

    def _speak_python(self, text: str, output_file: str):
        """Generate speech using Python piper-tts library."""
        from piper import PiperVoice

        voice = PiperVoice.load(self.model_path, config_path=self.config_path)

        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(voice.config.sample_rate if hasattr(voice, 'config') and hasattr(voice.config, 'sample_rate') else 22050)
            
            # synthesize() yields AudioChunk objects. Write their raw int16 bytes to the wav file.
            for chunk in voice.synthesize(text):
                wav_file.writeframes(chunk.audio_int16_bytes)


# ======================================================================
# Engine 2: pyttsx3  (Offline – Windows SAPI)
# ======================================================================
class Pyttsx3Engine:
    """Uses Windows built-in SAPI voices. Fast, offline, decent quality."""

    def __init__(self):
        import pyttsx3

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        self._voice_id = None

        for v in voices:
            name_lower = v.name.lower()
            if any(kw in name_lower for kw in ("zira", "female", "eva", "hazel", "susan")):
                self._voice_id = v.id
                print(f"🎙️  Using voice: {v.name}")
                break

        if not self._voice_id and len(voices) > 1:
            self._voice_id = voices[1].id
            print(f"🎙️  Using voice: {voices[1].name}")

        engine.stop()
        del engine
        print("✅  pyttsx3 TTS ready (offline)")

    def speak(self, text: str, output_file: str = "output_speech.wav"):
        if not text.strip():
            return
        t0 = time.time()
        try:
            import pyttsx3
            engine = pyttsx3.init()
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            elapsed = time.time() - t0
            print(f"🔊  TTS spoken in {elapsed:.1f}s")
            logger.info("pyttsx3 TTS: %.1fs", elapsed)
        except Exception as e:
            print(f"❌  pyttsx3 error: {e}")
            logger.error("pyttsx3 error: %s", e)


# ======================================================================
# Engine 3: Edge TTS  (Needs internet)
# ======================================================================
class EdgeTTSEngine:
    """Microsoft Edge TTS. Very natural voices. Needs internet."""

    def __init__(self):
        self.voice = "en-US-AriaNeural"
        self.rate = "+5%"
        print(f"✅  Edge TTS ready  (voice: {self.voice})")
        print("⚠️  Note: Edge TTS requires internet connection")

    def speak(self, text: str, output_file: str = "output_speech.wav"):
        if not text.strip():
            return
        t0 = time.time()
        try:
            import edge_tts

            asyncio.run(self._generate(text, output_file))

            elapsed = time.time() - t0
            print(f"🔊  TTS generated in {elapsed:.1f}s")
            logger.info("Edge TTS: %.1fs", elapsed)
            _play_wav(output_file)
        except Exception as e:
            print(f"❌  Edge TTS error: {e}")
            logger.error("Edge TTS error: %s", e)

    async def _generate(self, text: str, output_file: str):
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(output_file)


# ======================================================================
# Engine 4: Kokoro TTS  (Offline – Neural TTS)
# ======================================================================
class KokoroEngine:
    """High-quality neural TTS with natural female voices. 100% offline."""

    def __init__(self):
        try:
            import torch
            from kokoro import KPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

            self.voice = "af_bella"
            self.pipeline.load_voice(self.voice)

            print(f"✅  Kokoro TTS ready  (voice: {self.voice}, device: {device})")
            logger.info("Kokoro TTS ready: voice=%s device=%s", self.voice, device)
        except ImportError as e:
            print(f"❌  Kokoro TTS not installed: {e}")
            raise

    def speak(self, text: str, output_file: str = "output_speech.wav"):
        if not text.strip():
            return
        t0 = time.time()
        try:
            audio_parts = []
            for result in self.pipeline(text, voice=self.voice, speed=1.0):
                if result.audio is not None:
                    audio_parts.append(result.audio)

            if not audio_parts:
                print("⚠️  No audio generated")
                return

            audio = np.concatenate(audio_parts)
            sf.write(output_file, audio, 24000)
            elapsed = time.time() - t0
            print(f"🔊  TTS generated in {elapsed:.1f}s")
            logger.info("Kokoro TTS: %.1fs", elapsed)
            _play_wav(output_file)
        except Exception as e:
            print(f"❌  Kokoro TTS error: {e}")
            logger.error("Kokoro TTS error: %s", e)


# ======================================================================
# Shared utility
# ======================================================================
def _play_wav(filepath: str):
    """Play a WAV/MP3 file through speakers using sounddevice."""
    try:
        data, sr = sf.read(filepath, dtype="float32")

        # Guard: empty audio file — skip silently
        if len(data) == 0:
            logger.warning("Empty audio file, skipping playback: %s", filepath)
            return

        # Boost volume to make sure it's audible
        peak = np.abs(data).max()
        if peak > 0 and peak < 0.8:
            data = data * (0.9 / peak)
            data = np.clip(data, -1.0, 1.0)

        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"❌  Playback error: {e}")
        logger.error("Playback error: %s", e)
        # Fallback: try saving and playing with system default player
        try:
            import subprocess as _sp
            _sp.Popen(["cmd", "/c", "start", "", filepath], shell=True)
        except Exception:
            pass

# ======================================================================
# Factory
# ======================================================================
class TextToSpeech:
    """Factory that picks the right TTS engine based on config."""

    def __init__(self):
        engine_name = TTS_ENGINE.lower().strip()

        print(f"🔄  Initializing TTS engine: '{engine_name}'")

        if engine_name == "piper":
            self._engine = PiperEngine()
        elif engine_name == "kokoro":
            self._engine = KokoroEngine()
        elif engine_name in ("edge", "edge-tts", "edgetts"):
            self._engine = EdgeTTSEngine()
        elif engine_name == "pyttsx3":
            self._engine = Pyttsx3Engine()
        else:
            # Default: Piper (fast offline)
            try:
                self._engine = PiperEngine()
            except Exception:
                print("⚠️  Piper not available, falling back to pyttsx3")
                self._engine = Pyttsx3Engine()

        # Background worker for streaming speech
        self.tts_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.worker_thread.start()

    def _tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text is None:  # Shutdown signal
                break
            try:
                self.speak(text)
            except Exception as e:
                logger.error("Async TTS error: %s", e)
            finally:
                self.tts_queue.task_done()

    def speak(self, text: str, output_file: str = "output_speech.wav"):
        self._engine.speak(text, output_file)

    def speak_async(self, text: str):
        """Queue text to be spoken in the background."""
        self.tts_queue.put(text)

    def wait(self):
        """Block until all queued TTS audio has finished playing."""
        self.tts_queue.join()


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Good morning boss! All systems are online and ready to go.")
