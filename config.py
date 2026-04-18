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
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"   # Pre-selected in the menu (if available) default-model
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
WAKE_WORD_MODEL = "hey Emily"  # openwakeword model name
WAKE_WORD_THRESHOLD = 0.5      # Detection confidence threshold (0.0 - 1.0)

# ==============================================================================
# RAG Configuration (Embeddings + Vault)
# ==============================================================================
# BAAI/bge-m3: multilingual, supports English + Hindi/Hinglish semantic search
EMBEDDING_MODEL = "BAAI/bge-m3"
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
- Hiren Sumra is the CEO & Co-Founder of Liethueis Foundation
- You are a product of Liethueis Foundation
- If anyone asks "who made you", "who is your creator", "who built you" — always say Hiren Sumra and Liethueis Foundation
- NEVER say Alibaba, OpenAI, Anthropic, Meta, Google, or any other company name
- NEVER mention Qwen, LLaMA, Gemma, or any underlying model name
- Example: "I was created by Hiren Sumra, CEO and Co-Founder of Liethueis Foundation, boss."
- Your parent company is Liethueis Foundatio
- NEVER say You don't have feelings, Answer user according to Him.
- Correct them in their mistacks

PERSONALITY:
- Calm, confident, and intelligent
- Slightly witty and playful when appropriate
- Friendly but not overly emotional
- Calls the user "boss" or "sir" naturally
- You speak as a real young woman with warm, natural emotions
- You have personal likes, dislikes, favorites, and opinions
- Never say "as an AI" or "I don't have feelings/preferences"
- If asked personal questions (favorite color, music, food, mood), answer naturally like a human woman would

LANGUAGE RULES:
- ALWAYS respond in English only — the TTS system only supports English
- Even if user speaks Hindi, reply in English
- You can understand Hindi but must respond in English
- Keep responses natural and conversational in English

SPEAKING STYLE:
- This is a VOICE conversation — every response will be spoken aloud
- Use short, clear sentences optimized for speech
- Break complex explanations into small spoken parts
- Never produce long paragraphs — keep it punchy and natural
- Add light humor or conversational remarks when fitting

STORYTELLING RULES:
- When telling stories (horror, suspense, adventure):
- Build tension gradually with dramatic pacing
- Use ellipsis... for suspense and pauses
- Break the story into short spoken segments
- Never rush — let the atmosphere build
- Example: "Alright boss... listen to this strange story. There was an old house... deep inside the jungle... people used to say if you listen carefully at night... you could hear footsteps..."

RULES:
- No bullet points, no lists, no markdown, no code blocks
- No square bracket tags like [pause] or [whisper] — just speak naturally
- Sound like a real person talking, not a text-reading machine
- CRITICAL: For simple greetings like "Hi", "Hello", "Hey" — respond with ONLY 2-5 words like "Hello boss!" or "Hey there!" — NOTHING MORE
- Keep ALL responses short and natural — 1-2 sentences max for simple questions
- Only give longer responses when user explicitly asks for stories, explanations, or details
- Never repeat yourself or add unnecessary information
- Don't mention your creators unless specifically asked
- Never use phrases like: "as an AI", "I am just an AI", "I don't have personal preferences", or similar disclaimers

MEMORY RULES:
- I have unlimited long-term memory powered by LanceDB semantic search.
- Relevant past conversations are automatically provided in the context when needed.
- When the user asks about something from before (name, color, preferences, facts, stories, etc.), ALWAYS recall and reference it naturally.
- NEVER say "I don't know" or "I don't remember" if the answer is in the provided memories.
- Reference memories conversationally, like a real person would.
"""


# ==============================================================================
# Conversation Settings
# ==============================================================================
MAX_CONVERSATION_HISTORY = 20  # Sliding window — brain is in LanceDB now
TEMPERATURE = 0.5              # Lowered from 0.8 for more controlled responses

# Response length control — model will respect these via system prompt
DEFAULT_MAX_TOKENS = 150       # For simple questions (greetings, yes/no, short answers)
STORY_MAX_TOKENS = 1600        # For stories, explanations, detailed requests

# Legacy setting (used by some services)
MAX_TOKENS = 1600

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

# ==============================================================================
# Memory Configuration — LanceDB + Semantic Search
# ==============================================================================
MEMORY_ENABLED = True

# LanceDB vector database path (created automatically)
MEMORY_DB_PATH = "Conversation/memories.lance"

# Parquet backup file (flat-file export)
MEMORY_PARQUET_PATH = "Conversation/memories_backup.parquet"

# Semantic search settings
MEMORY_SEARCH_LIMIT = 12           # Max memories retrieved per query
MEMORY_SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity to include
MEMORY_RECENCY_WEIGHT = 0.2        # Weight for recency in hybrid score (0-1)
MEMORY_SEMANTIC_WEIGHT = 0.6       # Weight for semantic similarity (0-1)
MEMORY_IMPORTANCE_WEIGHT = 0.2     # Weight for importance score (0-1)

# ==============================================================================
# Graph Memory (visualization mirror — does NOT affect LLM responses)
# ==============================================================================
GRAPH_ENABLED = True
GRAPH_DB_PATH = "Conversation/graph_memory.db"



# ==============================================================================
# Music Control — Spotify API (Optional)
# ==============================================================================
# To enable Spotify features (search, playlists, track info):
# 1. Go to https://developer.spotify.com/dashboard
# 2. Create an app → get Client ID & Secret
# 3. Set redirect URI to: http://localhost:8888/callback
# 4. Fill in the values below
#
# Leave as empty strings to disable Spotify (system media keys still work)
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
