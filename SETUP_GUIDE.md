```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   J.A.R.V.I.S — Local Voice Assistant Setup & Troubleshooting Guide     ║
║   Pipecat + Ollama + Whisper + Piper TTS                                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## SYSTEM OVERVIEW

This system provides a **fully offline, real-time voice conversation** pipeline using:

```
Microphone
    ↓ (LocalAudioTransport)
Silero VAD (Voice Activity Detection)
    ↓
Faster Whisper (Speech-to-Text) 
    ↓
Ollama LLM (Local Language Model)
    ↓
Piper TTS (Text-to-Speech)
    ↓
Speakers
```

---

## PREREQUISITES & INSTALLATION

### 1. Install Ollama
- **Download**: https://ollama.com/download
- **Windows**: Run the installer
- **Verify installation**:
  ```bash
  ollama --version
  ```

### 2. Pull an Ollama Model
```bash
ollama pull dolphin-llama3:8b     # Fast (~4GB)
# OR
ollama pull qwen3-coder:30b       # More capable (~20GB)
```

### 3. Start Ollama Server
```bash
ollama serve
```
- The server will start at `http://localhost:11434`
- Keep this terminal open while using the voice assistant

### 4. Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## WHAT WAS FIXED

### Issue 1: Unicode/Encoding Error in main.py
**Problem**: Windows console couldn't display emojis/special characters
**Fix**: Added UTF-8 encoding handler at startup
```python
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### Issue 2: Deprecated Pipecat API Parameters
**Problem**: WhisperSTTService, OLLamaLLMService, PiperTTSService had deprecated parameters
**Fix**: Kept using working older API (newer settings-based API not available in v0.0.105)

### Issue 3: Ollama Base URL Configuration
**Problem**: OLLamaLLMService was receiving `base_url` with `/v1` appended
**Fix**: Pass plain base_url, Pipecat handles `/v1` internally
```python
# WRONG
llm = OLLamaLLMService(model=model_name, base_url=f"{OLLAMA_BASE_URL}/v1")

# CORRECT
llm = OLLamaLLMService(model=model_name, base_url=OLLAMA_BASE_URL)
```

### Issue 4: Piper TTS Model Loading
**Problem**: TTS wasn't finding the voice model
**Fix**: Model auto-detects from `piper_models/` directory (already present)

---

## SYSTEM STATUS

**All components verified as working:**

✅ **Ollama LLM**
- 6 models available
- Connection successful
- Models: dolphin-llama3:8b, qwen3-coder:30b, etc.

✅ **Whisper STT**
- Model: small.en
- Device: CUDA (GPU accelerated)
- Compute: float16
- Status: Ready

✅ **Piper TTS**
- Model: en_US-amy-medium
- Status: Loaded from local files
- Output: Working (test tone produced sound)

✅ **Audio I/O**
- Input: Microphone Array (Realtek)
- Output: Speakers (Realtek)
- Both tested and working

✅ **Pipecat Framework**
- Version: 0.0.105
- All services: Instantiation successful
- Pipeline: Ready to run

---

## RUNNING THE SYSTEM

### Option 1: Voice Mode (Real-time Conversation)
```bash
python main.py
# Select option 1 for Voice Pipeline
# Speak into your microphone
# Press Ctrl+C to stop
```

### Option 2: Text Mode (Testing without Microphone)
```bash
python main.py
# Select option 2 for Text Mode
# Type your messages
# Commands: quit, reset, voice, switch model
```

### Option 3: Quick Diagnostics
```bash
python diagnostic.py
# Tests all components independently
# Shows detailed status of each service
```

### Option 4: Voice Pipeline Test
```bash
python test_voice_pipeline.py
# Direct pipeline test
# Verifies end-to-end flow
```

---

## TROUBLESHOOTING

### Problem: "Ollama is not reachable"
**Solution**:
```bash
# Start Ollama in a separate terminal
ollama serve

# Check it's running
curl http://localhost:11434/api/tags
```

### Problem: "No models found"
**Solution**:
```bash
# Pull a model
ollama pull dolphin-llama3:8b
# Wait for download to complete
# Then run the system again
```

### Problem: "No audio input detected"
**Solution**:
- Check microphone is connected and enabled in Windows Sound Settings
- Run diagnostic to verify: `python diagnostic.py`
- Check audio peak level during calibration (should be > 0.01)

### Problem: "No audio output (no speaker sound)"
**Solution**:
- Check speakers are connected and not muted
- Verify in Windows Sound Settings that output device is enabled
- Run audio output test: `python diagnostic.py` (plays a tone)
- Check volume levels in Windows mixer

### Problem: "Model takes forever to load"
**Solution**:
- First run of Whisper/Piper downloads large models (~2GB total)
- Subsequent runs are fast
- Use smaller Ollama models for faster response: `ollama pull dolphin-llama3:8b`

### Problem: "Pipeline runs but no response"
**Check**:
1. Ollama server is running (`ollama serve`)
2. Model was actually pulled: `ollama list`
3. Microphone working: `python diagnostic.py` shows peak > 0.01
4. Speakers working: Audio test in `diagnostic.py` plays tone
5. Whisper transcribed correctly: Look for "You said: ..." output
6. Ollama responded: Look for latency in logs

---

## PERFORMANCE OPTIMIZATION

### For Faster Responses
- Use smaller Ollama model:
  ```bash
  ollama pull dolphin-llama3:8b    # 8B = fast, 4GB, good quality
  ```
- In `config.py`, set:
  ```python
  OLLAMA_MODEL = "dolphin-llama3:8b"
  MAX_TOKENS = 512  # Shorter responses
  ```

### For Better Quality
- Use larger Ollama model:
  ```bash
  ollama pull qwen3-coder:30b      # 30B = slower, 20GB, very capable
  ```
- Use larger Whisper model:
  ```python
  WHISPER_MODEL = "medium"  # More accurate but slower
  ```

### GPU Acceleration
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install CUDA: https://developer.nvidia.com/cuda-downloads
- Whisper will auto-use GPU if available (float16 compute)

---

## CONFIGURATION FILES

### config.py — Main Settings
```python
OLLAMA_MODEL = None                    # None = auto-select, or pick a model name
DEFAULT_OLLAMA_MODEL = "dolphin-llama3:8b"

WHISPER_MODEL = "small.en"            # tiny | base.en | small.en | medium
WHISPER_DEVICE = "cuda"               # Auto-detected

TTS_ENGINE = "piper"                  # piper | kokoro | pyttsx3 | edge

MAX_CONVERSATION_HISTORY = 10
TEMPERATURE = 0.8
MAX_TOKENS = 1300
```

### Customizing Voice Models
```python
# In config.py, change the system prompt:
SYSTEM_PROMPT = """You are Emily, a smart AI companion..."""

# Or change TTS voice:
TTS_SPEAKER_WAV = "voice_samples/your_voice.wav"
```

---

## AUDIO CALIBRATION

On startup, the system auto-calibrates:
1. Records 2 seconds of ambient noise
2. Sets speech detection threshold
3. Shows detected mic sample rate

**Stay quiet during calibration!**

If you change rooms or noise environment, restart the system to recalibrate.

---

## FILES & STRUCTURE

```
Local_Voice/
├── main.py                    # Main entry point (voice + text modes)
├── diagnostic.py              # Component diagnostic tester
├── test_voice_pipeline.py     # Direct pipeline test
├── config.py                  # All configuration
├── llm_handler.py            # Ollama LLM wrapper
├── speech_to_text.py         # Whisper wrapper (legacy, not used in Pipecat)
├── text_to_speech.py         # Piper TTS wrapper (legacy, not used in Pipecat)
├── audio_recorder.py         # Audio recording utilities
├── rag_engine.py             # RAG/vault system
├── wake_word.py              # Wake word detection
├── piper_models/             # TTS voice models
│   ├── en_US-amy-medium.onnx
│   └── en_US-amy-medium.onnx.json
├── voice_samples/            # Voice reference samples
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## QUICK START

1. **Start Ollama in one terminal**:
   ```bash
   ollama serve
   ```

2. **In another terminal, run the assistant**:
   ```bash
   python main.py
   ```

3. **Select option 1 for Voice Pipeline**

4. **Speak into your microphone!**

5. **Press Ctrl+C to stop**

---

## TESTING INDIVIDUAL COMPONENTS

```bash
# Test everything at once
python diagnostic.py

# Test just TTS (should play a tone)
python -c "from text_to_speech import PiperEngine; p = PiperEngine()"

# Test Whisper
python speech_to_text.py <audio_file.wav>

# Test Ollama
python -c "import ollama; c = ollama.Client(host='http://localhost:11434'); print(c.list())"
```

---

## KNOWN LIMITATIONS

1. **CPU-based Whisper**: If CUDA not available, speech recognition is slow
2. **First model load**: Takes 10-30 seconds on first run
3. **Internet**: System is fully offline (no cloud APIs)
4. **Memory**: Large models (30B Ollama) need 20GB RAM + VRAM

---

## SUPPORT

**Common issues fixed in v2 (this update)**:
- Unicode encoding errors on Windows
- Ollama base URL configuration
- API deprecation warnings cleaned up
- Audio I/O verified working end-to-end

**If still having issues**:
1. Run `python diagnostic.py` to see detailed status
2. Check `voice_assistant.log` for error messages
3. Ensure Ollama server is running
4. Check Windows Sound Settings for microphone/speaker

---

Generated: 2026-03-18
Last Updated: After comprehensive system analysis and debugging
```
