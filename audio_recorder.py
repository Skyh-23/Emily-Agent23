"""
Audio Recorder with Silero VAD + Energy-Based Voice Activity Detection
======================================================================
- Auto-calibrates the noise floor at startup
- Uses Silero VAD for precise speech boundary detection (P7)
- Falls back to energy-based VAD if Silero unavailable
- Imports SILENCE_THRESHOLD from config (P12)
- Has a maximum recording timeout to prevent hanging
"""

import sounddevice as sd
import numpy as np
import wave
import time
import logging
import torch
from config import SAMPLE_RATE, CHANNELS, SILENCE_DURATION, SILENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Try loading Silero VAD model
_silero_model = None
_silero_utils = None
try:
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    logger.info("Silero VAD loaded")
except Exception as e:
    logger.warning("Silero VAD not available, using energy-based VAD: %s", e)


class AudioRecorder:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE   # 16kHz target for Whisper/VAD
        self.channels = CHANNELS
        self.recording = []
        self.is_recording = False
        # P12: Use SILENCE_THRESHOLD from config as starting default
        self.threshold = SILENCE_THRESHOLD
        self.use_silero = _silero_model is not None

        # Device settings — None means system default (most compatible)
        self.input_device = None
        self.device_sample_rate = None  # Will query native rate if needed

        # Auto-calibrate on startup
        self._calibrate()

        if self.use_silero:
            print("✅  Audio recorder ready (Silero VAD)")
        else:
            print("✅  Audio recorder ready (energy-based VAD)")

    def _get_working_sample_rate(self, device=None):
        """
        Find a sample rate that works with the given device.
        Tries 16000 first (ideal for Whisper), then common rates.
        Returns the working sample rate.
        """
        rates_to_try = [16000, 44100, 48000, 22050, 8000]

        for rate in rates_to_try:
            try:
                # Test if this rate works
                sd.check_input_settings(device=device, samplerate=rate, channels=self.channels)
                logger.info("Sample rate %d Hz works for device %s", rate, device)
                return rate
            except Exception:
                continue

        # Last resort — query the device's default rate
        try:
            dev_info = sd.query_devices(device, kind='input')
            native_rate = int(dev_info['default_samplerate'])
            logger.info("Using device native rate: %d Hz", native_rate)
            return native_rate
        except Exception:
            return 16000  # Hope for the best

    def _calibrate(self):
        """Record 2 seconds of ambient noise to set the speech threshold."""
        print("🔧  Calibrating microphone... (stay quiet for 2 seconds)")

        # Find a working sample rate
        self.device_sample_rate = self._get_working_sample_rate(self.input_device)

        if self.device_sample_rate != self.sample_rate:
            print(f"🎙️  Mic native rate: {self.device_sample_rate}Hz → resampling to {self.sample_rate}Hz")
        else:
            print(f"🎙️  Mic sample rate: {self.device_sample_rate}Hz")

        try:
            samples = sd.rec(
                int(2 * self.device_sample_rate),
                samplerate=self.device_sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.input_device,
            )
            sd.wait()

            # Use RMS (root mean square) for better noise estimation
            rms = np.sqrt(np.mean(samples ** 2))

            # Threshold = 2x RMS of silence
            self.threshold = min(rms * 2.0, 0.015)
            self.threshold = max(self.threshold, 0.0005)

            print(f"✅  Mic calibrated!  noise_rms={rms:.6f}  threshold={self.threshold:.6f}")
            logger.info("Mic calibrated: rms=%.6f threshold=%.6f rate=%d",
                         rms, self.threshold, self.device_sample_rate)
        except Exception as e:
            print(f"⚠️  Calibration failed: {e}  —  using default threshold {SILENCE_THRESHOLD}")
            self.threshold = SILENCE_THRESHOLD
            logger.warning("Calibration failed: %s", e)

    def _resample(self, audio, from_rate, to_rate):
        """Simple linear resampling from one rate to another."""
        if from_rate == to_rate:
            return audio
        ratio = to_rate / from_rate
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        if audio.ndim == 1:
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        # Multi-channel
        result = np.zeros((n_samples, audio.shape[1]), dtype=np.float32)
        for ch in range(audio.shape[1]):
            result[:, ch] = np.interp(indices, np.arange(len(audio)), audio[:, ch])
        return result


    def record_until_silence(self, output_file="temp_recording.wav", max_duration=15):
        """
        Record audio until silence is detected after speech.
        Uses Silero VAD if available, otherwise falls back to energy-based detection.
        """
        if self.use_silero:
            return self._record_with_silero_vad(output_file, max_duration)
        else:
            return self._record_with_energy_vad(output_file, max_duration)

    # ------------------------------------------------------------------
    # P7: Silero VAD recording
    # ------------------------------------------------------------------
    def _record_with_silero_vad(self, output_file="temp_recording.wav", max_duration=15):
        """Record using Silero VAD for precise speech boundary detection."""
        print("\n🎤  Listening... (speak now, max {0}s)".format(max_duration))

        rec_rate = self.device_sample_rate or self.sample_rate
        self.recording = []
        speech_detected = False
        silence_start = None
        start_time = time.time()

        # Scale chunk size for recording rate (Silero wants 512 @ 16kHz = 32ms)
        chunk_size = int(512 * rec_rate / 16000)

        def callback(indata, frames, time_info, status):
            nonlocal speech_detected, silence_start

            if time.time() - start_time > max_duration:
                raise sd.CallbackStop()

            # Resample chunk to 16kHz for Silero VAD
            mono = indata[:, 0].copy()
            if rec_rate != 16000:
                ratio = 16000 / rec_rate
                n = int(len(mono) * ratio)
                mono_16k = np.interp(np.linspace(0, len(mono)-1, n), np.arange(len(mono)), mono).astype(np.float32)
            else:
                mono_16k = mono

            audio_chunk = torch.from_numpy(mono_16k)

            try:
                speech_prob = _silero_model(audio_chunk, 16000).item()
            except Exception:
                speech_prob = 1.0 if np.abs(indata).mean() > self.threshold else 0.0

            if speech_prob > 0.5:
                if not speech_detected:
                    speech_detected = True
                    print("🔴  Speech detected! Recording...")
                    logger.info("Speech detected (Silero VAD)")
                silence_start = None
                self.recording.append(indata.copy())
            elif speech_detected:
                if silence_start is None:
                    silence_start = time.time()
                self.recording.append(indata.copy())

                elapsed_recording = time.time() - start_time
                if elapsed_recording > 1.0 and time.time() - silence_start > SILENCE_DURATION:
                    raise sd.CallbackStop()

        try:
            with sd.InputStream(
                callback=callback,
                channels=self.channels,
                samplerate=rec_rate,
                dtype="float32",
                blocksize=chunk_size,
                device=self.input_device,
            ):
                while True:
                    sd.sleep(100)
                    if time.time() - start_time > max_duration + 2:
                        break
        except sd.CallbackStop:
            pass

        _silero_model.reset_states()

        if not self.recording:
            print("⚠️  No speech detected.")
            return None

        audio = np.concatenate(self.recording, axis=0)
        # Resample to 16kHz for Whisper
        audio = self._resample(audio, rec_rate, self.sample_rate)
        self._save_wav(output_file, audio)
        duration = len(audio) / self.sample_rate
        print(f"✅  Captured {duration:.1f}s of audio.")
        logger.info("Captured %.1fs with Silero VAD", duration)
        return output_file

    # ------------------------------------------------------------------
    # Energy-based VAD fallback
    # ------------------------------------------------------------------
    def _record_with_energy_vad(self, output_file="temp_recording.wav", max_duration=15):
        """Record audio until silence is detected using energy-based VAD."""
        print("\n🎤  Listening... (speak now, max {0}s)".format(max_duration))

        rec_rate = self.device_sample_rate or self.sample_rate
        self.recording = []
        silence_start = None
        recording_started = False
        start_time = time.time()
        threshold = self.threshold
        smoothed_vol = 0.0

        def callback(indata, frames, time_info, status):
            nonlocal silence_start, recording_started, smoothed_vol

            volume = np.abs(indata).mean()
            smoothed_vol = 0.3 * smoothed_vol + 0.7 * volume

            if time.time() - start_time > max_duration:
                raise sd.CallbackStop()

            if smoothed_vol > threshold:
                if not recording_started:
                    recording_started = True
                    print("🔴  Speech detected! Recording...")
                silence_start = None
                self.recording.append(indata.copy())
            elif recording_started:
                if silence_start is None:
                    silence_start = time.time()
                self.recording.append(indata.copy())

                elapsed_recording = time.time() - start_time
                if elapsed_recording > 1.0 and time.time() - silence_start > SILENCE_DURATION:
                    raise sd.CallbackStop()

        try:
            with sd.InputStream(
                callback=callback,
                channels=self.channels,
                samplerate=rec_rate,
                dtype="float32",
                blocksize=int(rec_rate * 0.1),
                device=self.input_device,
            ):
                while True:
                    sd.sleep(100)
                    if time.time() - start_time > max_duration + 2:
                        break
        except sd.CallbackStop:
            pass

        if not self.recording:
            print("⚠️  No speech detected.")
            return None

        audio = np.concatenate(self.recording, axis=0)
        audio = self._resample(audio, rec_rate, self.sample_rate)
        self._save_wav(output_file, audio)
        duration = len(audio) / self.sample_rate
        print(f"✅  Captured {duration:.1f}s of audio.")
        logger.info("Captured %.1fs with energy VAD", duration)
        return output_file

    def record_push_to_talk(self, output_file="temp_recording.wav"):
        """Hold SPACEBAR to record, release to stop."""
        import keyboard

        rec_rate = self.device_sample_rate or self.sample_rate

        print("\n🎤  Hold SPACEBAR to talk...")
        keyboard.wait("space")
        print("🔴  Recording...")

        self.recording = []
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.recording.append(indata.copy())

        stream = sd.InputStream(
            callback=callback,
            channels=self.channels,
            samplerate=rec_rate,
            dtype="float32",
            device=self.input_device,
        )
        stream.start()

        keyboard.wait("space")  # wait for release
        self.is_recording = False
        stream.stop()
        stream.close()

        if not self.recording:
            return None

        audio = np.concatenate(self.recording, axis=0)
        audio = self._resample(audio, rec_rate, self.sample_rate)
        self._save_wav(output_file, audio)
        duration = len(audio) / self.sample_rate
        print(f"✅  Captured {duration:.1f}s of audio.")
        return output_file

    def record_fixed_duration(self, output_file="temp_recording.wav", duration=5):
        """Record for a fixed number of seconds — simplest, always works."""
        rec_rate = self.device_sample_rate or self.sample_rate

        print(f"\n🎤  Recording for {duration} seconds... SPEAK NOW!")

        audio = sd.rec(
            int(duration * rec_rate),
            samplerate=rec_rate,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
        )
        sd.wait()

        audio = self._resample(audio, rec_rate, self.sample_rate)
        self._save_wav(output_file, audio)
        print(f"✅  Captured {duration}s of audio.")
        return output_file

    # ------------------------------------------------------------------
    def _save_wav(self, filename, data):
        """Save numpy float32 array → 16-bit WAV."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    rec = AudioRecorder()
    print("\nTest 1: Fixed duration recording")
    rec.record_fixed_duration("test_fixed.wav", 3)

    print("\nTest 2: VAD recording")
    rec.record_until_silence("test_vad.wav")
