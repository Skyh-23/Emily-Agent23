"""
Diagnostic Script — Test all components of the Voice Pipeline
==============================================================
Tests: Ollama, Whisper, Piper TTS, Audio I/O
"""

import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import json
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    TTS_ENGINE
)

def test_section(name: str):
    """Print a test section header."""
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print('='*70)

def success(msg: str):
    """Print success message."""
    print(f"  [✓] {msg}")

def error(msg: str):
    """Print error message."""
    print(f"  [✗] {msg}")

def info(msg: str):
    """Print info message."""
    print(f"  [*] {msg}")

# ======================================================================
# Test 1: Ollama Connection
# ======================================================================
test_section("Ollama Connection & Model Discovery")

try:
    import ollama
    info(f"Ollama Python SDK loaded successfully")
    
    client = ollama.Client(host=OLLAMA_BASE_URL)
    models = client.list().models
    
    if models:
        success(f"Ollama reachable! Found {len(models)} model(s):")
        for m in models:
            tag = " (CONFIGURED)" if hasattr(m, 'model') and m.model == OLLAMA_MODEL else ""
            print(f"    - {m.model}{tag}")
    else:
        error("Ollama server reachable but NO MODELS found")
        error(f"Pull a model with: ollama pull {DEFAULT_OLLAMA_MODEL}")
        
except Exception as e:
    error(f"Cannot connect to Ollama: {e}")
    error(f"Make sure Ollama is running: ollama serve")

# ======================================================================
# Test 2: Whisper STT
# ======================================================================
test_section("Faster Whisper (STT)")

try:
    from faster_whisper import WhisperModel
    
    info(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    success(f"Whisper loaded! (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE})")
    
except Exception as e:
    error(f"Whisper loading failed: {e}")

# ======================================================================
# Test 3: Piper TTS
# ======================================================================
test_section("Piper TTS (Text-to-Speech)")

try:
    from text_to_speech import PiperEngine
    
    info("Initializing Piper TTS engine...")
    piper = PiperEngine()
    
    if piper.piper_cmd:
        success(f"Piper CLI found: {piper.piper_cmd}")
    if piper.model_path:
        success(f"Piper model found: {piper.model_path}")
    if piper.config_path:
        print(f"    Config: {piper.config_path}")
    
    if not piper.piper_cmd or not piper.model_path:
        info("Piper will try to download model on first use")
        
except Exception as e:
    error(f"Piper TTS initialization failed: {e}")

# ======================================================================
# Test 4: Audio Input/Output
# ======================================================================
test_section("Audio I/O (sounddevice)")

try:
    import sounddevice as sd
    import numpy as np
    
    # List devices
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    
    success(f"sounddevice loaded - devices available:")
    print(f"    Default Input Device: #{default_in} - {devices[default_in]['name']}")
    print(f"    Default Output Device: #{default_out} - {devices[default_out]['name']}")
    
    # Test input
    try:
        info("Testing audio input (recording 1 second)...")
        audio = sd.rec(16000, samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        peak = np.abs(audio).max()
        success(f"Audio input working! Peak level: {peak:.3f}")
    except Exception as e:
        error(f"Audio input test failed: {e}")
    
    # Test output
    try:
        info("Testing audio output (playing 1 second tone)...")
        duration = 0.5
        freq = 440
        t = np.linspace(0, duration, int(16000 * duration), False)
        tone = 0.1 * np.sin(2 * np.pi * freq * t).astype('float32')
        sd.play(tone, 16000)
        sd.wait()
        success(f"Audio output working!")
    except Exception as e:
        error(f"Audio output test failed: {e}")
        
except ImportError as e:
    error(f"sounddevice not installed: {e}")
except Exception as e:
    error(f"Audio I/O test failed: {e}")

# ======================================================================
# Test 5: Pipecat Framework
# ======================================================================
test_section("Pipecat Framework")

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.transports.local.audio import LocalAudioTransport
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.piper.tts import PiperTTSService
    
    success("All Pipecat core components imported successfully")
    
    # Try to instantiate services
    try:
        stt = WhisperSTTService(
            model=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        success("WhisperSTTService instantiated")
    except Exception as e:
        error(f"WhisperSTTService failed: {e}")
    
    try:
        llm = OLLamaLLMService(
            model=DEFAULT_OLLAMA_MODEL,  # Use default if not configured
            base_url=OLLAMA_BASE_URL,
        )
        success("OLLamaLLMService instantiated")
    except Exception as e:
        error(f"OLLamaLLMService failed: {e}")
    
    try:
        tts = PiperTTSService(
            voice_id="en_US-amy-medium",
        )
        success("PiperTTSService instantiated")
    except Exception as e:
        error(f"PiperTTSService failed: {e}")
        
except ImportError as e:
    error(f"Pipecat import failed: {e}")
except Exception as e:
    error(f"Pipecat test failed: {e}")

# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC COMPLETE")
print('='*70)
print("\nRESULTS:")
print("  - If all tests passed, the system is ready to use")
print("  - If Ollama test failed, start Ollama: ollama serve")
print("  - If Whisper test failed, check CUDA availability")
print("  - If audio tests failed, check microphone and speaker connections")
print("  - If audio output test played a tone, your speaker works!")
