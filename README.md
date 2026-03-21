<div align="center">

![Liethueis Foundation]<img src="preview-image.png" width="300" height="264"/>

# Emily — Local Offline AI Agent

**Meet Emily. Your fully offline AI Agent that controls your PC — by voice.**

*Built by [Hiren Sumra](https://github.com/Skyh-23) — CEO & Co-Founder, Liethueis Foundation*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Offline](https://img.shields.io/badge/100%25-Offline-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

</div>

---

## 🧠 What is Emily?

**Emily** is a **fully offline AI Agent** that listens to your voice and takes real action on your computer — no cloud, no API keys, no internet after setup.

This is not a chatbot. This is not a voice assistant that plays music and sets timers.

This is an **AI Agent** — she understands your intent and executes real tasks:
opening files, editing code, controlling your PC, and soon, seeing your gestures.

> *"We are building the real JARVIS — but make it Emily. Fully offline, fully personal, fully yours."*
> — Hiren Sumra, Liethueis Foundation

---

## 🏗️ Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       EMILY PIPELINE                             │
│                                                                  │
│  🎤 Microphone                                                   │
│       │                                                          │
│       ▼                                                          │
│  🔍 Silero VAD  ←── detects speech, filters silence             │
│       │                                                          │
│       ▼                                                          │
│  📝 Faster Whisper (CUDA)  ←── Hindi + English, auto-detect     │
│       │                                                          │
│       ├────────────────────────────────┐                        │
│       │                                │                        │
│       ▼                                ▼                        │
│  🧠 Ollama LLM (local)        📚 RAG Memory Engine              │
│  qwen3-coder / dolphin        SQLite + BGE Embeddings           │
│       │                                │                        │
│       └────────────────────────────────┘                        │
│                       │                                          │
│                       ▼                                          │
│             🛠️  Agent Tool System                                │
│        open files · run commands · edit code                     │
│        create files · control apps · search vault               │
│                       │                                          │
│                       ▼                                          │
│            🔊 Kokoro Neural TTS                                  │
│       Hindi (hf_alpha) + English Indian accent (af_bella)       │
│                       │                                          │
│                       ▼                                          │
│                  🔈 Speakers                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ What Works Right Now

| Capability | Status | Details |
|-----------|--------|---------|
| 🎤 Real-time Voice Pipeline | ✅ Live | Pipecat streaming, sub-second latency |
| 🗣️ Hindi + English STT | ✅ Live | Faster Whisper CUDA, auto language detect |
| 🧠 Local LLM | ✅ Live | Ollama — qwen3-coder, dolphin, any model |
| 🔊 Neural TTS Hindi | ✅ Live | Kokoro hf_alpha — natural Hindi voice |
| 🔊 Neural TTS English | ✅ Live | Kokoro af_bella — Indian accent |
| 📚 RAG Memory | ✅ Live | SQLite vault, persistent across sessions |
| 🛠️ Open Applications | ✅ Live | Open any app by voice |
| 💻 Run Terminal Commands | ✅ Live | Execute shell commands by voice |
| 📄 Create / Edit Files | ✅ Live | Create and write files by voice |
| 🔀 Smart Model Routing | ✅ Live | Auto-picks best model per task type |
| 📄 PDF Ingestion | ✅ Live | Ingest documents into memory vault |

---

## 🚀 The Vision — Full PC Agent

```
TODAY                          FUTURE
─────────────────────          ──────────────────────────────────
✅ Voice commands              🔜 Full PC control by voice
✅ Open applications           🔜 Open/close/move any file or folder
✅ Run terminal commands        🔜 Browse folders, play mp3/mp4/images
✅ Create and edit files        🔜 Edit and save code by voice
✅ Hindi + English voice        🔜 Generate entire projects by voice
✅ RAG memory vault             🔜 Hand gesture — volume control
                               🔜 Hand gesture — brightness control
                               🔜 Eye tracking input
                               🔜 Screen understanding (vision)
                               🔜 Multi-agent task delegation
                               🔜 Android companion app
```

### 🖐️ Gesture Control (In Development)

```
Hand raised       →  Volume up
Hand lowered      →  Volume down
Pinch gesture     →  Brightness control
Two fingers up    →  Scroll up
Two fingers down  →  Scroll down
Fist              →  Pause / Stop
```

### 🗂️ Full File System Control (In Development)

```
"Open my projects folder"
"Play the last video in Downloads"
"Edit main.py in VS Code"
"Save this as utils.py"
"Open the mp3 in Music folder"
```

### 💻 Code Agent (In Development)

```
"Generate a script to rename all files in this folder"
"Fix the bug in line 42 of main.py"
"Add error handling to this function"
"Run the tests and tell me what failed"
```

---

## ⚙️ Configuration — One File Controls Everything

```python
# config.py — change one line to switch anything

OLLAMA_MODEL = None                    # None = model picker at startup
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"

WHISPER_MODEL = "small"               # tiny | small | medium | large-v3

TTS_ENGINE = "kokoro"                 # kokoro | piper | pyttsx3 | edge

MAX_CONVERSATION_HISTORY = 10
```

---

## 🗣️ Voice Commands (Current)

| Say This | What Happens |
|----------|-------------|
| *"Hey Emily, open YouTube"* | Opens YouTube in browser |
| *"Emily, open VS Code"* | Launches VS Code |
| *"Run pip install numpy"* | Executes in terminal |
| *"Create a file called test.py"* | Creates the file |
| *"Insert info [text]"* | Saves to memory vault |
| *"Print info"* | Reads back memory |
| *"Switch model"* | Change LLM at runtime |

---

## 📁 Project Structure

```
emily/
├── main.py               # Pipecat pipeline — entry point
├── config.py             # All settings — single source of truth
├── llm_handler.py        # Ollama + smart model routing
├── text_to_speech.py     # Kokoro dual-pipeline TTS
├── speech_to_text.py     # Faster Whisper STT
├── audio_recorder.py     # Silero VAD + mic recording
├── rag_engine.py         # SQLite vault + BGE embeddings
├── commands.py           # Agent tool system
├── wake_word.py          # Wake word detection
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/HirenSumra/emily-agent
cd emily-agent

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

ollama pull qwen3-coder:30b

python main.py
```

---

## 🤝 Why We're Applying for Claude API Access

Emily is built as a **fully offline agent** — but Claude would unlock capabilities that no local model can match:

**1. Agentic Code Generation**
User says *"build me a script that organizes my downloads folder by file type"* — Claude writes the code, Emily executes it. No IDE needed.

**2. Screen Understanding**
Claude Vision would let Emily read the screen, understand context, and act — seeing exactly what the user sees.

**3. Complex Multi-Step Tasks**
*"Find all my Python projects with no README and write one for each"* — Claude plans it, Emily executes step by step.

**4. Intelligent Fallback**
When local models struggle with complex reasoning, Claude handles it seamlessly — the user never knows the difference.

We are not building a demo. Emily is working, active, and used daily. Claude would help us build something genuinely new — a truly intelligent offline-first PC agent.

---

## 🗺️ Full Roadmap

### Phase 1 — Voice Agent ✅ Complete
- Real-time voice pipeline
- Hindi + English TTS
- Local LLM integration
- RAG memory system
- Basic tool system

### Phase 2 — PC Control 🔄 In Progress
- Full file system navigation
- Open/close/play any file type (mp3, mp4, folders, images)
- Code editor integration
- Advanced terminal control

### Phase 3 — Gesture & Vision 📋 Planned
- Hand gesture recognition (volume, brightness, scroll)
- Screen understanding via vision model
- Eye tracking input

### Phase 4 — Autonomous Agent 📋 Planned
- Multi-step task planning and execution
- Code generation + live execution
- Android companion app
- Multi-agent delegation

---

## 👩‍💻 About Liethueis Foundation

**Liethueis Foundation** is building the future of personal AI — private, powerful, and accessible to everyone.

| | |
|--|--|
| **Founder & CEO** | Hiren Sumra |
| **Mission** | Democratize AI — offline, private, open |
| **Focus** | AI Agents, Voice Interfaces, Gesture Control, PC Automation |
| **Philosophy** | Your AI should run on YOUR machine, not someone else's server |

---

## 📄 License

MIT License — free to use, modify, and build upon.

---

<div align="center">

**Emily — by Liethueis Foundation**

*intelligence. engineered.*

</div>
