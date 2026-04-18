# Emily — Setup Guide

Complete installation and configuration guide for Emily, the local offline AI voice agent.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **Python** | 3.10 | 3.12+ |
| **RAM** | 8 GB | 16 GB+ |
| **VRAM** | — | 6 GB+ (CUDA GPU) |
| **Disk** | 10 GB | 30 GB (for larger models) |
| **Microphone** | Any USB/built-in | — |
| **Speakers** | Any | — |

---

## Step 1: Install Ollama

Ollama runs LLMs locally on your machine.

1. **Download** from [ollama.com/download](https://ollama.com/download)
2. **Install** — run the Windows installer
3. **Verify**:
   ```bash
   ollama --version
   ```

### Pull a Model

```bash
# Fast & lightweight (~4GB) — good for quick responses
ollama pull dolphin-llama3:8b

# More capable (~20GB) — recommended for best quality
ollama pull qwen3-coder:30b
```

> You can pull multiple models. Emily lets you choose at startup.

---

## Step 2: Clone & Setup Python Environment

```bash
git clone https://github.com/AiDe-HirenSumra/Local_Voice.git
cd Local_Voice

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Image Generation

```bash
pip install -r image_gen_requirements.txt
```

### Optional: Faster Fuzzy Matching

```bash
pip install python-Levenshtein
```

---

## Step 3: First Run

### Terminal 1 — Start Ollama Server

```bash
ollama serve
```

Keep this terminal open. Ollama serves at `http://localhost:11434`.

### Terminal 2 — Start Emily

```bash
python main.py
```

### First Run Sequence

```
1. Embedding model downloads (~1GB, BAAI/bge-m3) — one time only
2. Mode selection: Voice Pipeline (1) or Text Mode (2)
3. Ollama model selection: choose from pulled models
4. Pipeline initializes (~10-30 seconds)
5. 🎤 Ready — speak into your microphone!
```

---

## Step 4: Verify Everything Works

### Quick System Check

```bash
python diagnostic.py
```

This tests:
- ✅ Ollama connection & available models
- ✅ Whisper STT initialization
- ✅ Piper TTS voice model
- ✅ Audio input/output devices
- ✅ Memory database connection

### Check Audio Devices

```bash
python check_audio.py
```

Verify your microphone is listed as the default input device.

---

## Configuration

All settings are in [`config.py`](config.py). Here are the most important ones:

### LLM Settings

```python
OLLAMA_MODEL = None                    # None = show picker, or set a specific model
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"  # Default selection in picker
OLLAMA_BASE_URL = "http://localhost:11434"
```

### Whisper STT

```python
WHISPER_MODEL = "small"          # Options: tiny, base.en, small, small.en, medium, large-v3
WHISPER_DEVICE = "cuda"          # Auto-detected: cuda (GPU) or cpu
WHISPER_COMPUTE_TYPE = "float16" # float16 (GPU) or int8 (CPU)
```

### TTS

```python
TTS_ENGINE = "piper"             # piper (recommended), kokoro, pyttsx3, edge
```

### Memory

```python
MEMORY_ENABLED = True
MEMORY_DB_PATH = "Conversation/memories.lance"
MEMORY_SEARCH_LIMIT = 12        # Max memories per query
MEMORY_SIMILARITY_THRESHOLD = 0.3
```

### Graph Visualization

```python
GRAPH_ENABLED = True
GRAPH_DB_PATH = "Conversation/graph_memory.db"
```

### Conversation

```python
TEMPERATURE = 0.5               # 0 = focused, 1 = creative
MAX_TOKENS = 1600               # Max response length
MAX_CONVERSATION_HISTORY = 20   # Sliding window size
```

### Spotify (Optional)

```python
SPOTIFY_CLIENT_ID = ""          # Leave empty to disable
SPOTIFY_CLIENT_SECRET = ""
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
```

---

## GPU Acceleration (Recommended)

Emily benefits significantly from CUDA GPU acceleration:

### Check CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Install CUDA-enabled PyTorch

If CUDA is not detected:

```bash
# Uninstall CPU-only torch
pip uninstall torch torchvision torchaudio

# Install CUDA 12.1 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### What Uses GPU

| Component | GPU Benefit |
|-----------|-------------|
| Whisper STT | **5-10x faster** transcription |
| LLM (Ollama) | **Automatic** — Ollama uses GPU if available |
| Embedding model | **2-3x faster** memory search |
| Piper TTS | CPU-only (already fast) |

---

## Document Ingestion

### Upload Documents to Memory

Emily can ingest PDFs, TXT, DOCX, and MD files into her LanceDB memory:

```bash
# Ingest a PDF
python upload_pdf.py path/to/document.pdf

# Ingest a text file
python upload_pdf.py notes.txt

# List all ingested documents
python upload_pdf.py --list

# Remove a document from memory
python upload_pdf.py --delete document.pdf
```

After ingestion, Emily can answer questions about the document content naturally:
> "What does the report say about Q2 revenue?"

---

## Graph Memory

Emily automatically builds a knowledge graph from conversations:

```bash
# Starts automatically with main.py
# Or run manually:
python graph_ui_server.py

# View the graph:
# Open http://127.0.0.1:8010 in your browser
```

### One-Time Backfill

If you want to populate the graph from existing memories:

```bash
python graph_backfill.py
```

> The graph is a **read-only visualization** — it never affects Emily's responses.

---

## Troubleshooting

### "Ollama is not reachable"

```bash
# Start Ollama in a separate terminal
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
# or open http://localhost:11434 in browser
```

### "No models found"

```bash
ollama pull dolphin-llama3:8b
ollama list   # verify models are available
```

### "No audio input detected"

1. Check Windows Sound Settings → Input → Microphone is enabled
2. Make sure mic is not muted in Windows privacy settings
3. Run `python check_audio.py` to see detected devices
4. Ensure no other app has exclusive mic access

### "Pipeline runs but Emily doesn't respond"

1. Check Ollama is running: `ollama serve`
2. Look for `"User started speaking"` in terminal output
3. If not appearing, microphone may be muted or VAD threshold too high
4. Check `voice_assistant.log` for detailed error messages

### "Slow responses"

1. Use a smaller model: `ollama pull dolphin-llama3:8b`
2. Set in config: `WHISPER_MODEL = "tiny"` (faster but less accurate)
3. Enable GPU: verify CUDA is available
4. Close other GPU-heavy applications

### "Memory not recalling properly"

1. Check memories exist: `python upload_pdf.py --list`
2. Verify LanceDB: check `Conversation/memories.lance/` directory exists
3. Lower threshold: set `MEMORY_SIMILARITY_THRESHOLD = 0.2` in config.py
4. Increase limit: set `MEMORY_SEARCH_LIMIT = 20`

### "Gesture control not working"

```bash
pip install mediapipe opencv-python pycaw screen-brightness-control
```

Ensure webcam is accessible and not used by another application.

---

## File Reference

| File | Purpose |
|------|---------|
| `main.py` | Entry point — voice pipeline + text mode + all interceptors |
| `config.py` | All configuration settings |
| `llm_handler.py` | LLM orchestration, intent routing, response generation |
| `intent_engine.py` | Fast keyword-based intent classifier (8 intents) |
| `commands.py` | App/web/system command executor |
| `custom_commands.py` | User-defined command macros |
| `file_opener.py` | Folder browser + fuzzy file search |
| `music_control.py` | Media key control + optional Spotify |
| `gesture_control.py` | Hand gesture volume/brightness |
| `image_gen.py` | AI image generation pipeline |
| `memory_store.py` | LanceDB memory CRUD + vector search |
| `memory_search.py` | Hybrid ranking + context injection |
| `embedding_model.py` | Shared BGE-M3 embedding model (singleton) |
| `rag_engine.py` | SQLite RAG knowledge vault |
| `upload_pdf.py` | Document → LanceDB ingestion tool |
| `graph_store.py` | SQLite graph node/edge storage |
| `graph_extractor.py` | Entity/relationship extraction |
| `graph_bridge.py` | Async LanceDB → graph sync |
| `graph_backfill.py` | One-time graph population script |
| `graph_ui_server.py` | Flask server for graph web UI |
| `text_to_speech.py` | TTS engines for text mode |
| `speech_to_text.py` | Whisper wrapper for text mode |
| `audio_recorder.py` | Audio recording utilities |
| `wake_word.py` | Wake word detection (not yet integrated) |
| `diagnostic.py` | System diagnostic tester |
| `check_audio.py` | Audio device listing utility |

---

## Requirements

### Core (required)

```
pipecat-ai[local,silero,whisper]>=0.0.105
ollama>=0.4.4
faster-whisper>=1.2.1
piper-tts>=1.2.0
torch>=2.0.0
sentence-transformers>=2.2.2
lancedb>=0.15.0
pyarrow>=14.0.0
sounddevice, soundfile, numpy, colorama
```

### Automation (required)

```
pyautogui, pywin32, psutil, fuzzywuzzy, pygetwindow
```

### Optional

```
PyPDF2          — PDF ingestion
python-docx     — DOCX ingestion
spotipy         — Spotify integration
opencv-python   — Gesture control
mediapipe       — Hand tracking
pycaw           — Volume control
openwakeword    — Wake word detection
```

---

## Updating

```bash
cd Local_Voice
git pull
pip install -r requirements.txt --upgrade
python main.py
```

---

## About

**Built by [Hiren Sumra](https://github.com/AiDe-HirenSumra)**
CEO & Co-Founder, **Liethueis Foundation**

Mission: Bring private, local-first AI to everyday users.

---

*Last updated: April 2026*
