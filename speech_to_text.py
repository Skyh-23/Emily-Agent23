"""
Speech-to-Text using Faster Whisper (with CUDA acceleration if available).
Includes audio amplification for quiet microphones.
Optimized for base.en model with GPU auto-detect (P3).
"""

import time
import logging
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
from pipecat.services.whisper.stt import WhisperSTTService

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(self):
        print(f"🔄  Loading Faster Whisper  model={WHISPER_MODEL}  device={WHISPER_DEVICE} ...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print(f"✅  Faster Whisper loaded! (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE})")
        logger.info("Whisper loaded: model=%s device=%s compute=%s",
                     WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)

    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file → text string.
        Auto-amplifies quiet audio for better recognition."""

        t0 = time.time()

        # Amplify quiet audio before transcribing
        amplified_file = self._amplify_if_needed(audio_file)

        segments, info = self.model.transcribe(
            amplified_file,
            beam_size=7,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=600,     # Don't cut on brief pauses
                speech_pad_ms=400,               # Extra padding around speech
                threshold=0.35,                  # Lower = more sensitive to speech
            ),
        )

        text = " ".join(seg.text for seg in segments).strip()
        elapsed = time.time() - t0

        if text:
            print(f'👤  You said: "{text}"')
            logger.info("STT: '%s' (%.2fs)", text[:100], elapsed)
        else:
            print("⚠️  Could not transcribe any speech.")
            logger.warning("STT: no speech detected (%.2fs)", elapsed)

        return text

    def _amplify_if_needed(self, audio_file: str) -> str:
        """If audio is too quiet, amplify it so Whisper can hear it."""
        try:
            data, sr = sf.read(audio_file, dtype="float32")
            peak = np.abs(data).max()

            if peak < 0.01:
                # Very quiet — skip, likely no speech
                return audio_file

            if peak < 0.5:
                # Normalize to 0.9 peak
                gain = 0.9 / peak
                data = data * gain
                # Clip to prevent distortion
                data = np.clip(data, -1.0, 1.0)
                sf.write(audio_file, data, sr)
                print(f"🔊  Audio amplified {gain:.1f}x for better recognition")
                logger.info("Audio amplified %.1fx", gain)

            return audio_file
        except Exception:
            return audio_file


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    stt = SpeechToText()
    if len(sys.argv) > 1:
        stt.transcribe(sys.argv[1])
    else:
        print("Usage: python speech_to_text.py <audio_file>")
