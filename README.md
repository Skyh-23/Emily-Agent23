<div align="center">

# Emily — Local Offline AI Voice Agent

### Voice-first personal AI for your computer
**Private · Fast · Local · Action-ready**

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Offline](https://img.shields.io/badge/100%25-Offline-brightgreen)
![Windows](https://img.shields.io/badge/Platform-Windows-0078D6)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900)
![Pipecat](https://img.shields.io/badge/Pipecat-0.0.105-purple)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## What is Emily?

Emily is a **fully local, offline AI voice agent** that can understand speech, reason with local LLMs, control your PC, manage files, remember past conversations, and respond with neural voice — all without ever sending data to the cloud.

```
🎤 You speak → 📝 Whisper transcribes → ⚡ Intent engine routes
     → 🧠 Ollama LLM reasons → 💾 Memory saves → 🔊 Piper speaks
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Emily Voice Pipeline                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🎤 Microphone                                                  │
│   └→ Silero VAD (voice activity detection)                      │
│       └→ Faster Whisper STT (speech → text)                     │
│           └→ ⚡ Interceptor Chain                                │
│               ├─ ImageRequestProcessor (AI image generation)    │
│               ├─ GestureRequestProcessor (hand gesture control) │
│               ├─ MusicRequestProcessor (media controls)         │
│               ├─ FolderBrowserProcessor (file navigation)       │
│               └─ AppRequestProcessor (open/close/type/search)   │
│                   └→ VoiceUserCaptureProcessor                  │
│                       └→ VoiceMemoryInjectProcessor             │
│                           └→ 🧠 Ollama LLM                     │
│                               └→ LLMOutputSanitizer             │
│                                   └→ MemoryFrameProcessor       │
│                                       └→ 🔊 Piper TTS          │
│                                           └→ 🔈 Speakers       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Background Services                                        │ │
│  │  📊 Graph Memory UI (http://127.0.0.1:8010)                │ │
│  │  💾 LanceDB + BGE-M3 Semantic Memory                       │ │
│  │  🕸️  Neural Graph (SQLite mirror for visualization)         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Voice Pipeline** | ✅ | Real-time streaming conversation via Pipecat |
| **Speech-to-Text** | ✅ | Faster Whisper (CUDA/CPU), multilingual |
| **LLM Reasoning** | ✅ | Any Ollama model — interactive selection at startup |
| **Text-to-Speech** | ✅ | Piper neural TTS (en_US-amy-medium) |
| **Semantic Memory** | ✅ | LanceDB + BGE-M3 — unlimited long-term recall |
| **Graph Memory** | ✅ | Entity/relationship extraction with web UI visualization |
| **RAG Vault** | ✅ | SQLite knowledge base with semantic retrieval |
| **Document Ingestion** | ✅ | PDF/TXT/DOCX → LanceDB memory via `upload_pdf.py` |
| **Intent Engine** | ✅ | Fast offline intent detection (<50ms, no LLM needed) |
| **App Control** | ✅ | Open, close, list apps — with Windows Search |
| **Typing Automation** | ✅ | Type text into any active window |
| **Web Search** | ✅ | Open websites, Google/YouTube search |
| **Music Control** | ✅ | Play/pause/next/prev/volume via system media keys |
| **Spotify** | ⚙️ | Optional Spotify API integration |
| **Folder Browser** | ✅ | Voice-navigable folder listing with numeric selection |
| **File Search** | ✅ | Fuzzy indexed file search with type filtering |
| **Gesture Control** | ✅ | Hand-based volume/brightness via MediaPipe |
| **Image Generation** | ✅ | AI prompt enhancement + Stable Diffusion |
| **Custom Commands** | ✅ | Self-learning command macros ("gaming mode" → open steam + discord) |
| **Compound Commands** | ✅ | Multi-step commands in one sentence |
| **Text Mode** | ✅ | Keyboard mode for testing without microphone |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Skyh-23/Local_Voice.git
cd Local_Voice

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install & start Ollama (separate terminal)
ollama pull qwen2.5-coder:7b   # or any model you prefer, gemma4:e4b for all agentic task
ollama serve

# 5. Run Emily
python main.py
```

> **First run** downloads ~2GB of models (Whisper, BGE-M3, Piper). Subsequent starts are fast (~10s).

---

## Modes

### 🎤 Voice Pipeline (Option 1)
Real-time conversation — speak into your microphone, Emily responds with voice.

### 💬 Text Mode (Option 2)
Keyboard input for testing. Same LLM + memory, but typed instead of spoken.

**Text mode commands:**
```
quit              — exit
reset             — clear conversation history
voice             — switch to voice mode
switch model      — change Ollama model
list models       — show available models
insert info <text> — add text to RAG vault
print info        — display vault contents
delete info       — clear vault
```

---

## Voice Commands

### App & Window Control
```
"open chrome"                    "close notepad"
"list apps"                      "open notepad and type hello world"
```

### Web
```
"open youtube"                   "search how to learn python"
"google best restaurants nearby"
```

### Music
```
"play music"    "pause"    "next song"    "previous song"
"volume up"     "volume down"    "mute"   "what's playing"
```

### Folder Browser
```
"open music folder"    "play 1"    "open 2"
"go back"              "close browser"
```

### Memory
```
"remember this — meeting at 5pm tomorrow"
"show memories"    "search memory birthday"
```

### Gesture Control
```
"gesture volume"       "gesture brightness"
"stop gesture"
```

### Image Generation
```
"generate image of a sunset over mountains"
"create image of a cat in a spaceship"
```

### Custom Commands
```
"when I say gaming mode, open steam and discord"
"gaming mode"  → executes saved actions instantly
```

---

## Document Memory

Ingest PDF, TXT, DOCX, or MD files directly into Emily's LanceDB memory:

```bash
# Ingest a document
python upload_pdf.py report.pdf

# List all ingested documents
python upload_pdf.py --list

# Delete a document's memories
python upload_pdf.py --delete report.pdf
```

After ingestion, just ask naturally:
> *"What does report.pdf say about revenue?"*

Emily recalls it automatically via semantic search.

---

## Graph Memory Visualization

Emily builds a neural graph of entities and relationships from every conversation:

```bash
# Auto-starts with main.py, or run manually:
python graph_ui_server.py
# Open: http://127.0.0.1:8010
```

The graph is a **read-only visualization mirror** — it never injects data into LLM responses.

---

## Project Structure

```
Local_Voice/
├── main.py                  # Entry point — voice pipeline + text mode
├── config.py                # All configuration (LLM, TTS, memory, etc.)
├── llm_handler.py           # Ollama LLM orchestration + intent routing
├── intent_engine.py         # Fast offline intent classifier (<50ms)
├── commands.py              # App/web/system automation executor
├── custom_commands.py       # Self-learning command macros
├── file_opener.py           # Folder browser + fuzzy file search
├── music_control.py         # Media keys + optional Spotify
├── gesture_control.py       # Hand gesture volume/brightness (MediaPipe)
├── image_gen.py             # AI image generation pipeline
│
├── memory_store.py          # LanceDB CRUD + semantic vector search
├── memory_search.py         # Hybrid memory ranking + context injection
├── embedding_model.py       # Shared BGE-M3 embedding singleton
├── rag_engine.py            # SQLite RAG knowledge vault
├── upload_pdf.py            # PDF/TXT/DOCX → LanceDB document memory
│
├── graph_store.py           # SQLite graph node/edge storage
├── graph_extractor.py       # Entity/relationship extraction from text
├── graph_bridge.py          # Async LanceDB → graph sync (fire-and-forget)
├── graph_backfill.py        # One-time backfill existing memories to graph
├── graph_ui_server.py       # Flask server for graph visualization
├── graph_ui/                # Web UI (HTML/JS/CSS) for graph explorer
│
├── text_to_speech.py        # Piper/pyttsx3/edge TTS engines (text mode)
├── speech_to_text.py        # Whisper STT wrapper (text mode)
├── audio_recorder.py        # Audio recording utilities
├── wake_word.py             # Wake word detection (openwakeword)
│
├── requirements.txt         # Core Python dependencies
├── image_gen_requirements.txt # Optional: Stable Diffusion deps
├── en_US-amy-medium.onnx    # Piper TTS voice model (~60MB)
├── Conversation/            # Runtime data directory
│   ├── memories.lance/      #   LanceDB vector database
│   ├── graph_memory.db      #   SQLite graph database
│   └── custom_commands.json #   Saved custom command macros
└── generated_images/        # AI-generated images output
```

---

## Configuration

All settings live in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `None` (auto-select) | Force a specific model, or `None` for startup menu |
| `DEFAULT_OLLAMA_MODEL` | `qwen3-coder:30b` | Pre-selected in menu if available |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base.en/small/medium/large-v3) |
| `WHISPER_DEVICE` | auto-detected | `cuda` if GPU available, else `cpu` |
| `TTS_ENGINE` | `piper` | TTS engine (piper/kokoro/pyttsx3/edge) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence embedding model for memory |
| `MEMORY_ENABLED` | `True` | Enable/disable long-term memory |
| `MEMORY_SEARCH_LIMIT` | `12` | Max memories retrieved per query |
| `GRAPH_ENABLED` | `True` | Enable/disable graph visualization mirror |
| `TEMPERATURE` | `0.5` | LLM temperature (0=focused, 1=creative) |
| `MAX_TOKENS` | `1600` | Maximum response length |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Ollama not reachable | Run `ollama serve` in a separate terminal |
| No mic input | Check Windows Sound Settings → microphone is enabled and not muted |
| Slow responses | Use a smaller model: `ollama pull dolphin-llama3:8b` |
| Memory recall weak | Check `Conversation/memories.lance` exists and has data |
| Gesture not working | Verify camera access and MediaPipe: `pip install mediapipe` |
| First run slow | Models download on first use (~2GB). Subsequent runs are fast |
| CUDA not detected | Install CUDA toolkit: `python -c "import torch; print(torch.cuda.is_available())"` |
| Pipeline runs but no speech | Check VAD is triggering (look for "User started speaking" in logs) |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Pipeline | Pipecat 0.0.105 | Real-time audio streaming framework |
| STT | Faster Whisper | Speech-to-text (CUDA accelerated) |
| LLM | Ollama | Local language model inference |
| TTS | Piper | Neural text-to-speech |
| VAD | Silero | Voice activity detection |
| Memory | LanceDB + PyArrow | Vector database for semantic memory |
| Embeddings | BAAI/bge-m3 | Multilingual sentence embeddings |
| Graph | SQLite | Entity/relationship graph storage |
| Graph UI | Flask + D3.js | Interactive graph visualization |
| Automation | pyautogui, pywin32, psutil | PC control |
| Gestures | MediaPipe + OpenCV | Hand tracking |

---

## About

**Built by [Hiren Sumra](https://github.com/Skyh-23)**
CEO & Co-Founder, **Liethueis Foundation**

Mission: Bring private, local-first AI to everyday users.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
