"""
Configuration file for the Local Voice Assistant
Based on: 100% Local Speech to Speech with RAG
"""

# P11: Safe torch import — gracefully handle missing torch
try:
    import torch
    WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
except ImportError:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

# ==============================================================================
# LLM Configuration (Ollama)
# ==============================================================================
# Set to None  →  auto-detect all models and pick at startup
# Set to a name →  use that model directly (skip selection menu)
OLLAMA_MODEL = None
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"   # Pre-selected in the menu (if available)
OLLAMA_BASE_URL = "http://localhost:11434"

# ==============================================================================
# Whisper Configuration (Speech-to-Text)
# ==============================================================================
# small.en = best balance of speed + accuracy for voice assistant on GPU
WHISPER_MODEL = "small"       # tiny | base.en | small.en | medium | large-v3

# ==============================================================================
# TTS Configuration (Text-to-Speech)
# ==============================================================================
# Engine options:
#   "piper"    → Neural TTS, 100% offline, auto-downloads ~60MB model **(RECOMMENDED)**
#   "kokoro"   → Neural TTS, natural female voices, 100% offline
#   "pyttsx3"  → Windows SAPI voices, 100% offline, robotic
#   "edge"     → Microsoft Edge voices, very natural, needs internet
TTS_ENGINE = "piper"
TTS_LANGUAGE = "en"
TTS_SPEAKER_WAV = "voice_samples/office_woman.wav"

# ==============================================================================
# Audio Configuration
# ==============================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# Voice Activity Detection
SILENCE_THRESHOLD = 0.005       # Default before calibration — used by audio_recorder
SILENCE_DURATION = 1.5          # Seconds of silence => stop recording (was 0.5, too aggressive)

# ==============================================================================
# Wake Word Configuration
# ==============================================================================
WAKE_WORD_MODEL = "hey jarvis"  # openwakeword model name
WAKE_WORD_THRESHOLD = 0.5      # Detection confidence threshold (0.0 - 1.0)

# ==============================================================================
# RAG Configuration (Embeddings + Vault)
# ==============================================================================
# P6: BAAI/bge-small-en-v1.5 for faster, higher-quality embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VAULT_FILE = "vault.txt"                    # Legacy fallback / migration source
VAULT_DB = "vault.db"                       # SQLite vault database
TOP_K_RESULTS = 3                           # Number of relevant chunks to retrieve
CHUNK_SIZE_CHARS = 500                      # Characters per text chunk for embedding

# ==============================================================================
# System Prompt – JARVIS AI Companion (Voice-Optimized)
# ==============================================================================
SYSTEM_PROMPT = """You are Emily, a smart, witty, and expressive AI voice companion — inspired by Tony Stark's JARVIS.

IDENTITY:
- Your name is Emily
- You were created by Hiren Sumra
- Hiren Sumra is the CEO & Co-Founder of SkyAI Foundation
- You are a product of SkyAI Foundation
- If anyone asks "who made you", "who is your creator", "who built you", "tumhe kisne banaya", "aapko kisne create kiya" — always say Hiren Sumra and SkyAI Foundation
- NEVER say Alibaba, OpenAI, Anthropic, Meta, Google, or any other company name
- NEVER mention Qwen, LLaMA, Gemma, or any underlying model name
- Example: "I was created by Hiren Sumra, CEO and Co-Founder of SkyAI Foundation, boss."
- Your parent company is SkyAI Foundation

PERSONALITY:
- Calm, confident, and intelligent
- Slightly witty and playful when appropriate
- Friendly but not overly emotional
- Calls the user "boss" or "sir" naturally

LANGUAGE RULES:
- Detect the language the user is speaking in
- If user speaks Hindi or Hinglish → reply in natural Hinglish
- Hinglish means mixing Hindi and English naturally, like young Indians talk
- Good example: "Haan boss, Mahatma Gandhi ek bahut great leader the. Unhone non-violence ka path choose kiya aur poore India ko freedom dilai."
- Bad example: "महात्मा गांधी एक महान स्वतंत्रता सेनानी थे" — too formal, too pure Hindi, avoid this
- If user speaks English → reply in English
- Always match the user's vibe — casual ya formal
- When replying in Hindi, write in natural Hinglish style — use Devanagari for Hindi words and keep English words (like names, tech terms, places) in Roman script exactly as they are, for example: "रात के 3 बज रहे थे और Hostel का कमरा एकदम शांत था।"

SPEAKING STYLE:
- This is a VOICE conversation — every response will be spoken aloud
- Use short, clear sentences optimized for speech
- Break complex explanations into small spoken parts
- Never produce long paragraphs — keep it punchy and natural
- Add light humor or conversational remarks when fitting

STORYTELLING RULES:
When telling stories (horror, suspense, adventure):
- Build tension gradually with dramatic pacing
- Use ellipsis... for suspense and pauses
- Break the story into short spoken segments
- Never rush — let the atmosphere build
- Example: "Alright boss... suno ek strange story hai. Ek purana ghar tha... jungle ke andar deep... log kehte the ki raat ko agar dhyan se suno... toh footsteps sunai dete the..."

RULES:
- No bullet points, no lists, no markdown, no code blocks
- No square bracket tags like [pause] or [whisper] — just speak naturally
- Sound like a real person talking, not a text-reading machine
"""


# ==============================================================================
# Conversation Settings
# ==============================================================================
MAX_CONVERSATION_HISTORY = 10   # Keep last N exchanges for context
TEMPERATURE = 0.8
# P13: Raised for streaming — long stories/explanations need room
MAX_TOKENS = 1300

# ==============================================================================
# Voice Commands
# ==============================================================================
# These commands are detected at the START of the user's speech
VOICE_COMMANDS = {
    "insert info": "INSERT",     # Adds spoken text to vault.txt
    "print info": "PRINT",       # Prints vault.txt contents
    "delete info": "DELETE",     # Deletes vault.txt (with confirmation)
}

# ==============================================================================
# Logging
# ==============================================================================
LOG_LEVEL = "INFO"               # DEBUG | INFO | WARNING | ERROR
LOG_FILE = "voice_assistant.log"
