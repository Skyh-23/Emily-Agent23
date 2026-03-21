"""
Wake Word Detection using openwakeword
=======================================
Listens for a wake word (default: "hey jarvis") and signals the main loop
to start recording. Minimal CPU usage while idle.
"""

import logging
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, WAKE_WORD_THRESHOLD

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Listens for a wake word using openwakeword."""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.threshold = WAKE_WORD_THRESHOLD
        self.model = None

        try:
            from openwakeword.model import Model as OWWModel

            # Initialize model — downloads on first use
            self.model = OWWModel(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            logger.info("Wake word detector initialized: hey_jarvis")
            print("✅  Wake word detector ready (say 'Hey Jarvis')")
        except ImportError:
            print("⚠️  openwakeword not installed. Install with: pip install openwakeword")
            print("    Wake word detection disabled — falling back to manual activation.")
            logger.warning("openwakeword not available")
        except Exception as e:
            print(f"⚠️  Wake word init error: {e}")
            print("    Wake word detection disabled — falling back to manual activation.")
            logger.error("Wake word init failed: %s", e)

    @property
    def available(self) -> bool:
        return self.model is not None

    def wait_for_wake_word(self) -> bool:
        """
        Block until wake word is detected.
        Returns True when wake word is detected, False on error.
        """
        if not self.model:
            return False

        print("🔇  Listening for wake word... (say 'Hey Jarvis')")
        detected = False

        # Process audio in 80ms chunks (1280 samples at 16kHz)
        chunk_size = 1280

        def callback(indata, frames, time_info, status):
            nonlocal detected
            if detected:
                raise sd.CallbackStop()

            # openwakeword expects int16
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

            # Feed to model
            prediction = self.model.predict(audio_int16)

            # Check all wake word models
            for model_name, score in prediction.items():
                if score > self.threshold:
                    detected = True
                    logger.info("Wake word detected: %s (score=%.3f)", model_name, score)
                    print(f"🎯  Wake word detected! (confidence: {score:.2f})")
                    raise sd.CallbackStop()

        try:
            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=chunk_size,
            ):
                while not detected:
                    sd.sleep(100)
        except sd.CallbackStop:
            pass
        except Exception as e:
            logger.error("Wake word listening error: %s", e)
            print(f"❌  Wake word error: {e}")
            return False

        # Reset model state for next detection
        if self.model:
            self.model.reset()

        return detected


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = WakeWordDetector()
    if detector.available:
        print("Say 'Hey Jarvis' to test...")
        if detector.wait_for_wake_word():
            print("✅  Wake word detected!")
        else:
            print("❌  No wake word detected")
    else:
        print("Wake word detection not available")
