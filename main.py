import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

"""
J.A.R.V.I.S  —  Local Voice Assistant (Pipecat Pipeline)
=========================================================
Fully offline, real-time streaming voice pipeline using Pipecat.
Pipeline:  Microphone → Silero VAD → Whisper STT → Ollama LLM → Piper TTS → Speaker
Usage:  python main.py
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from colorama import Fore, Style, init as colorama_init
colorama_init(strip=False)

BANNER = f"""
\033[36m==============================================================
=                                                            =
=   [MIC]  J.A.R.V.I.S  —  Pipecat Voice Pipeline         =
=         100% Offline  ·  Real-Time Streaming             =
=         Whisper STT  ·  Ollama LLM  ·  Piper TTS        =
=                                                            =
==============================================================\033[0m
"""
print(BANNER)
print(f"{Fore.YELLOW}[LOADING]  Loading AI Models and Modules... Please wait (10-30 seconds).{Style.RESET_ALL}\n")
sys.stdout.flush()

# ── Early load: Embedding model (used by RAG + Memory) ──
# Load this FIRST so it's ready when memories are accessed
from embedding_model import get_embedding_model
_embedding_model = get_embedding_model()  # Preload now, not lazily later

import asyncio
import signal
import logging
import threading
import time
import re
from typing import Optional
import ollama as _ollama

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.frames.frames import (
    TextFrame, TTSSpeakFrame, TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from image_gen import generate_image
from gesture_control import start_gesture, stop_gesture
from music_control import get_music_controller
from file_opener import (
    get_file_opener, is_browser_selection_command, 
    is_browser_navigation_command, get_browser_state
)

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS,
    LOG_LEVEL, LOG_FILE,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(name)-20s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _start_graph_ui_server_if_enabled():
    """Run Graph UI server in background thread (auto-start with main app)."""
    try:
        import config as _cfg
        if not getattr(_cfg, "GRAPH_ENABLED", True):
            return
        host = getattr(_cfg, "GRAPH_UI_HOST", "127.0.0.1")
        port = getattr(_cfg, "GRAPH_UI_PORT", 8010)
    except Exception:
        host, port = "127.0.0.1", 8010

    try:
        from graph_ui_server import run_server
    except Exception as e:
        logger.warning("Graph UI import failed: %s", e)
        return

    def _runner():
        try:
            run_server(host=host, port=port)
        except OSError as e:
            logger.warning("Graph UI server skipped (port busy): %s", e)
        except Exception as e:
            logger.warning("Graph UI server failed: %s", e)

    t = threading.Thread(target=_runner, name="GraphUIThread", daemon=True)
    t.start()
    print(f"\U0001f578\ufe0f  Graph UI: http://{host}:{port}")
    logger.info("Graph UI thread started at http://%s:%s", host, port)


# ======================================================================
# Image Request Interceptor
# ======================================================================
class ImageRequestProcessor(FrameProcessor):
    IMAGE_KEYWORDS = [
        "generate image", "create image", "make image",
        "draw", "image of", "picture of", "show me a picture",
        "image banao", "tasveer banao", "photo banao",
        "ek image", "generate a photo", "paint", "sketch",
    ]

    def __init__(self, ollama_client, model_name):
        super().__init__()
        self.client = ollama_client
        self.model  = model_name

    def _is_image_request(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.IMAGE_KEYWORDS)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and self._is_image_request(frame.text):
            print(f"🎨  Image request detected: {frame.text}")
            await self.push_frame(TTSSpeakFrame(
                "Sure boss, generating your image right now. Please wait a moment."
            ))
            result = generate_image(
                prompt=frame.text,
                llm_client=self.client,
                llm_model=self.model,
            )
            await self.push_frame(TTSSpeakFrame(result))
            return
        await self.push_frame(frame, direction)


# ======================================================================
# Gesture Request Interceptor — handles gesture commands in voice mode
# ======================================================================
class GestureRequestProcessor(FrameProcessor):
    """
    Intercepts gesture-related voice commands and executes them directly.
    Prevents these commands from going to the LLM.
    """
    
    # Stop gesture keywords - checked FIRST
    STOP_KEYWORDS = [
        "stop gesture", "gesture stop", "gesture off", "gesture band",
        "camera off", "camera band", "hand tracking off", "stop hand",
        "gesture band karo", "camera band karo",
        "close gesture", "close hand gesture", "exit gesture", "gesture close",
        "gesture exit", "hand gesture off", "hand gesture band",
        "close camera", "end gesture", "gesture end",
    ]
    
    def _get_gesture_mode(self, text: str) -> str | None:
        """
        Check if text is a gesture command.
        Returns: 'stop', 'volume', 'brightness', 'general', or None
        """
        lower = text.lower()
        
        # Check stop first - highest priority
        if any(kw in lower for kw in self.STOP_KEYWORDS):
            return "stop"
        
        # Check for BRIGHTNESS - look for "bright" anywhere with gesture context
        if "bright" in lower and ("gesture" in lower or "hand" in lower or "control" in lower):
            return "brightness"
        
        # Check for VOLUME - look for "volume" anywhere with gesture context
        if "volume" in lower and ("gesture" in lower or "hand" in lower or "control" in lower):
            return "volume"
        
        # General gesture mode - only if no specific mode mentioned
        general_kw = ["gesture mode", "gesture control", "hand detection",
                      "hand tracking", "camera control", "start gesture",
                      "haath dikhao", "gesture chalu karo", "gesture start"]
        if any(kw in lower for kw in general_kw):
            return "general"
        
        # Fallback: "hand gesture" alone (without brightness/volume) = general
        if "hand gesture" in lower and "bright" not in lower and "volume" not in lower:
            return "general"
        
        return None
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            # Debug: log what we received
            print(f"🔍 GestureProcessor received: '{frame.text}'")
            
            gesture_mode = self._get_gesture_mode(frame.text)
            
            if gesture_mode:
                print(f"🖐️  Gesture command detected: {frame.text} → mode: {gesture_mode}")
                
                if gesture_mode == "stop":
                    result = stop_gesture()
                else:
                    result = start_gesture(gesture_mode)
                
                # Send TTS response instead of passing to LLM
                await self.push_frame(TTSSpeakFrame(result))
                return  # Don't pass to next processor
        
        await self.push_frame(frame, direction)


# ======================================================================
# Music Request Interceptor — handles music commands in voice mode
# ======================================================================
class MusicRequestProcessor(FrameProcessor):
    """
    Intercepts music-related voice commands and executes them directly.
    Supports: play, pause, next, previous, volume, mute, status.
    """
    
    # Play music keywords
    PLAY_KEYWORDS = [
        "play music", "play song", "play the music", "resume music",
        "music play", "start music", "music chalu", "gaana chalu",
        "play my music", "continue music", "music on",
    ]
    
    # Pause music keywords
    PAUSE_KEYWORDS = [
        "pause music", "pause the music", "stop music", "music pause",
        "music stop", "music band", "gaana band", "pause song",
        "stop the music", "music off",
    ]
    
    # Next track keywords
    NEXT_KEYWORDS = [
        "next song", "next track", "skip song", "skip track",
        "agla gaana", "next music", "skip this", "next one",
    ]
    
    # Previous track keywords
    PREV_KEYWORDS = [
        "previous song", "previous track", "go back song", "last song",
        "pichla gaana", "previous music", "back track",
    ]
    
    # Volume keywords
    VOLUME_UP_KEYWORDS = ["volume up", "louder", "increase volume", "awaz badha"]
    VOLUME_DOWN_KEYWORDS = ["volume down", "quieter", "decrease volume", "awaz kam"]
    MUTE_KEYWORDS = ["mute music", "mute", "unmute", "mute sound"]
    
    # Status keywords
    STATUS_KEYWORDS = [
        "what's playing", "what is playing", "current song", "which song",
        "kya chal raha", "konsa gaana", "what song is this", "song name",
    ]
    
    def _get_music_action(self, text: str) -> str | None:
        """
        Check if text is a music command.
        Returns: 'play', 'pause', 'next', 'prev', 'vol_up', 'vol_down', 'mute', 'status', or None
        """
        lower = text.lower()
        
        # Check each category
        if any(kw in lower for kw in self.STATUS_KEYWORDS):
            return "status"
        if any(kw in lower for kw in self.NEXT_KEYWORDS):
            return "next"
        if any(kw in lower for kw in self.PREV_KEYWORDS):
            return "prev"
        if any(kw in lower for kw in self.PAUSE_KEYWORDS):
            return "pause"
        if any(kw in lower for kw in self.PLAY_KEYWORDS):
            return "play"
        if any(kw in lower for kw in self.VOLUME_UP_KEYWORDS):
            return "vol_up"
        if any(kw in lower for kw in self.VOLUME_DOWN_KEYWORDS):
            return "vol_down"
        if any(kw in lower for kw in self.MUTE_KEYWORDS):
            return "mute"
        
        return None
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            action = self._get_music_action(frame.text)
            
            if action:
                print(f"🎵 Music command detected: {frame.text} → action: {action}")
                
                # Get music controller (lazy load)
                ctrl = get_music_controller()
                
                # Execute the action
                if action == "play":
                    result = ctrl.play()
                elif action == "pause":
                    result = ctrl.pause()
                elif action == "next":
                    result = ctrl.next_track()
                elif action == "prev":
                    result = ctrl.previous_track()
                elif action == "vol_up":
                    result = ctrl.volume_up()
                elif action == "vol_down":
                    result = ctrl.volume_down()
                elif action == "mute":
                    result = ctrl.mute()
                elif action == "status":
                    result = ctrl.get_status()
                else:
                    result = "Sorry boss, I didn't understand that music command."
                
                # Send TTS response
                await self.push_frame(TTSSpeakFrame(result))
                return  # Don't pass to next processor
        
        await self.push_frame(frame, direction)


# ======================================================================
# Folder Browser Interceptor — handles browse folder and selection commands
# ======================================================================
class FolderBrowserProcessor(FrameProcessor):
    """
    Handles interactive folder browsing:
    - "open music folder" / "browse movies" → list contents with numbers
    - "play 1" / "open 2" → select item by number
    - "go back" / "close browser" → navigation
    """
    
    # Browse folder keywords (different from just "open folder")
    BROWSE_KEYWORDS = [
        "browse", "show", "list", "dikhao", "batao",
        "what's in", "kya hai", "open music folder", "open movies folder",
        "open videos folder", "open pictures folder", "open documents folder",
        "open downloads folder", "show music", "show movies",
    ]
    
    def __init__(self):
        super().__init__()
        self._file_opener = None
    
    def _get_opener(self):
        if self._file_opener is None:
            self._file_opener = get_file_opener()
        return self._file_opener
    
    def _is_browse_request(self, text: str) -> bool:
        """Check if this is a folder browse request."""
        lower = text.lower()
        
        # Check for browse keywords
        if any(kw in lower for kw in self.BROWSE_KEYWORDS):
            return True
        
        # Check if browser is active and this might be a selection
        state = get_browser_state()
        if state["active"]:
            is_sel, _ = is_browser_selection_command(text)
            if is_sel:
                return True
            if is_browser_navigation_command(text):
                return True
        
        return False
    
    def _extract_folder_name(self, text: str) -> Optional[str]:
        """Extract folder name from browse request."""
        lower = text.lower()
        
        # Known folder patterns
        folder_patterns = [
            "music folder", "music", "songs", "gaane",
            "movies folder", "movies", "films",
            "videos folder", "videos",
            "pictures folder", "pictures", "photos", "images",
            "documents folder", "documents", "docs",
            "downloads folder", "downloads",
            "desktop",
        ]
        
        for pattern in folder_patterns:
            if pattern in lower:
                return pattern
        
        return None
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        # Handle both TextFrame and TranscriptionFrame (voice input)
        text = None
        if isinstance(frame, TextFrame):
            text = frame.text
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text
        
        if text:
            opener = self._get_opener()
            state = get_browser_state()
            
            # Debug log browser state
            if state["active"]:
                print(f"📂 Browser active: {state['folder']} ({state['item_count']} items)")
            
            # Check if browser is active and this is a selection command
            if state["active"]:
                # Check for selection command (play 1, open 2, etc.)
                is_sel, num = is_browser_selection_command(text)
                if is_sel:
                    print(f"📂 Browser selection: {num}")
                    result = await opener.select_item(num)
                    await self.push_frame(TTSSpeakFrame(result))
                    return
                
                # Check for navigation command
                nav_cmd = is_browser_navigation_command(text)
                if nav_cmd == "back":
                    result = await opener.go_back()
                    await self.push_frame(TTSSpeakFrame(result))
                    return
                elif nav_cmd == "close":
                    result = opener.close_browser()
                    await self.push_frame(TTSSpeakFrame(result))
                    return
            
            # Check if this is a browse folder request
            folder_name = self._extract_folder_name(text)
            if folder_name and self._is_browse_request(text):
                print(f"📂 Browse folder request: {folder_name}")
                result, items = await opener.browse_folder(folder_name)
                await self.push_frame(TTSSpeakFrame(result))
                return
        
        await self.push_frame(frame, direction)


# ======================================================================
# App Request Interceptor — handles app open/close commands in voice mode
# ======================================================================
class AppRequestProcessor(FrameProcessor):
    """
    Intercepts app open/close/type/website voice commands and executes them directly.
    Supports compound commands like "open notepad and type hello".
    Uses Windows Search method (Win + type + Enter) for opening ANY app.
    """
    
    # Compound command separators
    COMPOUND_SEPARATORS = [
        " and then ", " then ", " and ", " aur phir ", " phir ", " aur ", 
        ", then ", ", and ", ", ", " uske baad ",
    ]
    
    # Close app keywords (check FIRST - more specific)
    CLOSE_KEYWORDS = [
        "close ", "band karo", "band kar", "exit ", "quit ", 
        "bund karo", "bund kar", "hatao", "shut down ",
        "close the", "band kardo",
    ]
    
    # Open app keywords
    OPEN_KEYWORDS = [
        "open ", "kholo", "khol do", "launch ", "start ", 
        "chalu karo", "chalu kar", "run ", "open the",
    ]
    
    # List apps keywords
    LIST_KEYWORDS = [
        "list apps", "running apps", "open apps", "kya chal raha",
        "konsi app", "kitni apps", "show apps", "active windows",
    ]
    
    # Type text keywords
    TYPE_KEYWORDS = [
        "type ", "likho ", "likh do ", "write ", "likho yeh ",
        "type this ", "likho ki ", "likh ", "type the text ",
        "into this ", "in this ", "isme ",
    ]
    
    # Keywords that indicate content should be GENERATED by LLM, not typed literally
    GENERATION_TRIGGERS = [
        "random", "generate", "create", "make", "banao", "bana do",
        "paragraph", "story", "poem", "essay", "letter", "email",
        "code", "function", "script", "program",
        "words", "lines", "sentences",
        "about", "on topic", "regarding", "ke baare me",
        "something", "kuch", "anything",
    ]
    
    # Website keywords
    WEBSITE_KEYWORDS = [
        "open website", "website kholo", "browser me kholo",
        "open site", "site kholo", "web kholo",
    ]
    
    # Search keywords
    SEARCH_KEYWORDS = [
        "search ", "google karo ", "google search ", "search for ",
        "google pe dhundo ", "search karo ", "dhundo ",
    ]
    
    # Common websites that should open directly (not as apps)
    KNOWN_WEBSITES = [
        "amazon", "flipkart", "youtube", "google", "facebook", "instagram",
        "twitter", "linkedin", "reddit", "netflix", "spotify", "github",
        "gmail", "whatsapp", "telegram", "wikipedia", "pinterest", "twitch",
        "stackoverflow", "chatgpt", "claude", "notion", "hotstar", "myntra",
    ]
    
    # Common Whisper ASR misheard corrections (misheard -> correct)
    WORD_CORRECTIONS = {
        "world": "word",       # "Close World" → "Close Word" (Microsoft Word)
        "worlds": "word",
        "node pad": "notepad",
        "no pad": "notepad", 
        "no bad": "notepad",
        "note pad": "notepad",
        "cod": "code",         # VS Code
        "scout": "spotify",
        "spotify": "spotify",  
        "one": "1",            # "Open one" → "Open 1"
        "first": "1",
        "two": "2",
        "second": "2",
        "three": "3",
        "third": "3",
        "four": "4",
        "fifth": "5",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    
    # File/folder keywords — route to file_opener instead of Windows Search
    FILE_FOLDER_KEYWORDS = [
        "folder", "directory", "song", "music folder", "video folder",
        "movies folder", "pictures folder", "documents folder", "downloads folder",
        "gaana", "gana", "photo", "pdf", "document",
        ".mp3", ".mp4", ".pdf", ".jpg", ".png", ".mkv",
    ]

    AMBIGUOUS_OPEN_TARGETS = {
        "music", "songs", "gaane",
        "movies", "films", "videos",
        "pictures", "photos", "images",
        "documents", "docs", "downloads",
    }
    
    def __init__(self):
        super().__init__()
        from commands import CommandExecutor
        import time
        self.executor = CommandExecutor()
        self._time = time
        self._ollama_client = None  # Lazy load for content generation
        self._file_opener = None  # Lazy load file opener
        self._intent_choice_cache = {}  # target -> "app" | "folder"
        self._pending_clarification = None  # {"target": "music"}
        self._selected_model = None  # Injected from voice pipeline
    
    def _get_file_opener(self):
        """Lazy load FileOpener instance."""
        if self._file_opener is None:
            from file_opener import get_file_opener
            self._file_opener = get_file_opener()
        return self._file_opener
    
    def _is_file_or_folder_request(self, target: str) -> bool:
        """Check if target looks like a file/folder request (not an app)."""
        target_lower = target.lower()
        return any(kw in target_lower for kw in self.FILE_FOLDER_KEYWORDS)

    def _normalize_target(self, target: str) -> str:
        """Normalize target text for cache/intent matching."""
        t = target.lower().strip().rstrip(".,!?")
        t = " ".join(t.split())
        return t

    def _extract_disambiguation_choice(self, text: str) -> Optional[str]:
        """Extract app/folder choice from a clarification reply."""
        lower = self._normalize_target(text)
        if any(w in lower for w in ["folder", "directory", "files"]):
            return "folder"
        if any(w in lower for w in ["app", "application", "software", "program"]):
            return "app"
        return None

    def _is_ambiguous_open_target(self, target: str) -> bool:
        """Check if target can reasonably mean app or folder."""
        t = self._normalize_target(target)
        if "folder" in t or "directory" in t:
            return False
        return t in self.AMBIGUOUS_OPEN_TARGETS
    
    def _get_ollama_client(self):
        """Lazy load Ollama client for content generation."""
        if self._ollama_client is None:
            try:
                from ollama import Client
                from config import OLLAMA_BASE_URL
                self._ollama_client = Client(host=OLLAMA_BASE_URL)
            except Exception as e:
                print(f"⚠️ Could not initialize Ollama client: {e}")
        return self._ollama_client
    
    def _needs_generation(self, text: str) -> bool:
        """Check if the text needs LLM generation (not literal typing)."""
        lower = text.lower()
        return any(trigger in lower for trigger in self.GENERATION_TRIGGERS)
    
    def _generate_content(self, prompt: str) -> str:
        """Generate content using LLM based on user's request."""
        client = self._get_ollama_client()
        if not client:
            return ""
        
        try:
            from config import OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL
            model_name = self._selected_model or OLLAMA_MODEL or DEFAULT_OLLAMA_MODEL
            
            # Create a focused prompt for content generation
            system_prompt = """You are a content generator. Generate ONLY the requested content.
Do NOT add explanations, introductions, or "Here's..." phrases.
Do NOT use markdown formatting.
Output ONLY the raw content that should be typed."""
            
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate this content (output ONLY the content, nothing else): {prompt}"}
                ],
                options={"temperature": 0.7}
            )
            
            content = response['message']['content'].strip()
            # Remove any markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return content
            
        except Exception as e:
            print(f"⚠️ Content generation failed: {e}")
            logger.warning("Content generation failed; falling back to original text.")
            return ""
    
    def _split_compound_command(self, text: str) -> list[str]:
        """
        Split compound command into individual commands.
        "open notepad and type hello" → ["open notepad", "type hello"]
        """
        lower = text.lower().strip()
        
        # Try each separator (longest first for proper matching)
        for sep in sorted(self.COMPOUND_SEPARATORS, key=len, reverse=True):
            if sep in lower:
                parts = lower.split(sep)
                # Clean up parts
                commands = [p.strip() for p in parts if p.strip()]
                if len(commands) > 1:
                    return commands
        
        # No compound separator found - single command
        return [lower]

    def _normalize_search_query(self, query: str) -> str:
        """Clean common filler words from search query."""
        q = query.strip()
        for prefix in ["for ", "about ", "on ", "regarding "]:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
        return q
    
    def _apply_word_corrections(self, text: str) -> str:
        """Apply ASR misheard word corrections."""
        words = text.split()
        corrected = []
        for word in words:
            word_lower = word.lower().rstrip(".,!?")
            if word_lower in self.WORD_CORRECTIONS:
                corrected.append(self.WORD_CORRECTIONS[word_lower])
            else:
                corrected.append(word)
        return " ".join(corrected)
    
    def _get_action(self, text: str) -> tuple[str, str]:
        """
        Check if text is an app/type/website command.
        Returns: (action, target) where action is 'open', 'close', 'list', 'type', 'website', 'search', or ''
        """
        # Apply ASR corrections first
        text = self._apply_word_corrections(text)
        lower = text.lower().strip()

        # Resolve pending clarification first ("folder" / "app")
        if self._pending_clarification:
            choice = self._extract_disambiguation_choice(text)
            if choice:
                target = self._pending_clarification["target"]
                self._pending_clarification = None
                self._intent_choice_cache[target] = choice
                if choice == "folder":
                    return "file_or_folder", target
                return "open", target
            return "clarify", self._pending_clarification["target"]
        
        # List running apps
        if any(kw in lower for kw in self.LIST_KEYWORDS):
            return "list", ""
        
        # Search Google
        for kw in self.SEARCH_KEYWORDS:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    for suffix in ["karo", "kar", "please", "boss"]:
                        query = query.replace(suffix, "").strip()
                    if query:
                        return "search", query
        
        # Type text (check before open to catch "type something")
        for kw in self.TYPE_KEYWORDS:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    text_to_type = parts[1].strip()
                    # Remove quotes if present
                    text_to_type = text_to_type.strip('"\'')
                    # Clean common suffixes
                    for suffix in ["into this", "in this", "isme", "please", "boss"]:
                        text_to_type = text_to_type.replace(suffix, "").strip()
                    if text_to_type:
                        # Check if this needs generation
                        if self._needs_generation(text_to_type):
                            return "generate_type", text_to_type
                        return "type", text_to_type
        
        # Close app (check before open - more specific)
        for kw in self.CLOSE_KEYWORDS:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    app_name = parts[1].strip()
                    # Clean common suffixes
                    for suffix in ["karo", "kar", "do", "please", "boss"]:
                        app_name = app_name.replace(suffix, "").strip()
                    if app_name:
                        return "close", app_name
        
        # Open website or app or file/folder
        for kw in self.OPEN_KEYWORDS:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    target = parts[1].strip()
                    # Clean common suffixes
                    for suffix in ["karo", "kar", "do", "please", "boss", "for me"]:
                        target = target.replace(suffix, "").strip()
                    
                    if target:
                        # Check if it's a known website
                        target_clean = target.lower().replace(" ", "")
                        for site in self.KNOWN_WEBSITES:
                            if site in target_clean or target_clean in site:
                                return "website", target
                        
                        # Check if it looks like a URL
                        if ".com" in target or ".in" in target or ".org" in target or "www." in target:
                            return "website", target
                        
                        # Check if it's a file/folder request
                        if self._is_file_or_folder_request(target):
                            return "file_or_folder", target

                        normalized_target = self._normalize_target(target)

                        # Apply cached decision for ambiguous targets
                        cached_choice = self._intent_choice_cache.get(normalized_target)
                        if cached_choice == "folder":
                            return "file_or_folder", normalized_target
                        if cached_choice == "app":
                            return "open", target

                        # Ask clarification for ambiguous open requests
                        if self._is_ambiguous_open_target(normalized_target):
                            self._pending_clarification = {"target": normalized_target}
                            return "clarify", normalized_target
                        
                        # Otherwise treat as app
                        return "open", target
        
        return "", ""
    
    async def _execute_action_async(self, action: str, target: str) -> str:
        """Execute a single action (async version for file operations)."""
        if action == "file_or_folder":
            # Use file_opener for files and folders
            opener = self._get_file_opener()
            result = await opener.smart_open(target)
            if result:
                return result
            # Fallback to app opener if file_opener didn't find anything
            return self.executor.open_application(target)
        elif action == "clarify":
            return (
                f"Boss, did you mean {target} folder or {target} app? "
                "Say 'folder' or 'app'."
            )
        else:
            # Sync actions
            return self._execute_action(action, target)
    
    def _execute_action(self, action: str, target: str) -> str:
        """Execute a single action and return result."""
        if action == "open":
            return self.executor.open_application(target)
        elif action == "file_or_folder":
            # Sync fallback — try file_opener sync, then app
            # Note: For async context, use _execute_action_async instead
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    opener = self._get_file_opener()
                    # Can't await here, fall back to app
                    return self.executor.open_application(target)
                else:
                    opener = self._get_file_opener()
                    result = loop.run_until_complete(opener.smart_open(target))
                    if result:
                        return result
                    return self.executor.open_application(target)
            except Exception:
                return self.executor.open_application(target)
        elif action == "close":
            return self.executor.close_application(target)
        elif action == "list":
            return self.executor.list_running_apps()
        elif action == "type":
            return self.executor.type_text(target)
        elif action == "generate_type":
            # Generate content using LLM, then type it
            print(f"✨ Generating content for: {target}")
            generated = self._generate_content(target)
            if generated and generated.strip():
                print(f"✨ Generated {len(generated)} chars, typing...")
                return self.executor.type_text(generated)
            print("⚠️ Generation unavailable, typing requested text directly.")
            return self.executor.type_text(target)
        elif action == "website":
            return self.executor.open_website(target)
        elif action == "search":
            return self.executor.search_web(target)
        elif action == "clarify":
            return (
                f"Boss, did you mean {target} folder or {target} app? "
                "Say 'folder' or 'app'."
            )
        else:
            return "Sorry boss, I didn't understand that command."
    
    def _execute_compound_command(self, text: str) -> tuple[bool, str]:
        """
        Execute compound command (e.g., "open notepad and type hello").
        Returns: (was_compound, result_message)
        """
        commands = self._split_compound_command(text)
        
        if len(commands) == 1:
            # Single command
            action, target = self._get_action(commands[0])
            if action:
                result = self._execute_action(action, target)
                return True, result
            return False, ""
        
        # Multiple commands - execute sequentially
        results = []
        print(f"🔗 Compound command detected: {len(commands)} parts")
        
        for i, cmd in enumerate(commands):
            action, target = self._get_action(cmd)
            if action:
                # Smart chain: open <site> + search <query> => search on that site
                if (
                    action == "search"
                    and i > 0
                    and i < len(commands)
                ):
                    prev_action, prev_target = self._get_action(commands[i - 1])
                    prev_site = prev_target.lower().strip()
                    if prev_action in ("open", "website") and prev_site in self.KNOWN_WEBSITES:
                        site_query = self._normalize_search_query(target)
                        result = self.executor.search_on_website(prev_site, site_query)
                        print(f"   [{i+1}] site_search: {prev_site} -> {site_query}")
                        results.append(result)
                        if i < len(commands) - 1:
                            self._time.sleep(0.5)
                        continue

                print(f"   [{i+1}] {action}: {target}")
                result = self._execute_action(action, target)
                results.append(result)
                
                # Wait between commands (especially after open)
                if action == "open" and i < len(commands) - 1:
                    self._time.sleep(2.0)  # Wait for app to open
                elif i < len(commands) - 1:
                    self._time.sleep(0.5)  # Small delay between other commands
        
        if results:
            # Return summary
            if len(results) == 1:
                return True, results[0]
            else:
                return True, f"Done boss! Executed {len(results)} commands."
        
        return False, ""
    
    async def _execute_compound_command_async(self, text: str) -> tuple[bool, str]:
        """
        Async version of compound command execution (for file/folder operations).
        """
        commands = self._split_compound_command(text)
        
        if len(commands) == 1:
            action, target = self._get_action(commands[0])
            if action:
                if action == "file_or_folder":
                    result = await self._execute_action_async(action, target)
                else:
                    result = self._execute_action(action, target)
                return True, result
            return False, ""
        
        # Multiple commands
        results = []
        print(f"🔗 Compound command detected: {len(commands)} parts")
        
        for i, cmd in enumerate(commands):
            action, target = self._get_action(cmd)
            if action:
                # Smart chain: open <site> + search <query> => search on that site
                if (
                    action == "search"
                    and i > 0
                    and i < len(commands)
                ):
                    prev_action, prev_target = self._get_action(commands[i - 1])
                    prev_site = prev_target.lower().strip()
                    if prev_action in ("open", "website") and prev_site in self.KNOWN_WEBSITES:
                        site_query = self._normalize_search_query(target)
                        result = self.executor.search_on_website(prev_site, site_query)
                        print(f"   [{i+1}] site_search: {prev_site} -> {site_query}")
                        results.append(result)
                        if i < len(commands) - 1:
                            await asyncio.sleep(0.5)
                        continue

                print(f"   [{i+1}] {action}: {target}")
                if action == "file_or_folder":
                    result = await self._execute_action_async(action, target)
                else:
                    result = self._execute_action(action, target)
                results.append(result)
                
                if action in ("open", "file_or_folder") and i < len(commands) - 1:
                    await asyncio.sleep(2.0)
                elif i < len(commands) - 1:
                    await asyncio.sleep(0.5)
        
        if results:
            if len(results) == 1:
                return True, results[0]
            else:
                return True, f"Done boss! Executed {len(results)} commands."
        
        return False, ""
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            was_command, result = await self._execute_compound_command_async(frame.text)
            
            if was_command:
                print(f"📱  Command executed: {frame.text}")
                await self.push_frame(TTSSpeakFrame(result))
                return  # Don't pass to next processor
        
        # Also check TranscriptionFrame (voice input)
        if isinstance(frame, TranscriptionFrame):
            was_command, result = await self._execute_compound_command_async(frame.text)
            
            if was_command:
                print(f"📱  Voice Command executed: {frame.text}")
                await self.push_frame(TTSSpeakFrame(result))
                return  # Don't pass to next processor
        
        await self.push_frame(frame, direction)


# ======================================================================
# Shared state for memory saving (between processors)
# ======================================================================
_voice_memory_state = {
    "last_user_text": "",
}


# ======================================================================
# Voice User Capture Processor — captures latest user voice text
# ======================================================================
class VoiceUserCaptureProcessor(FrameProcessor):
    """Captures latest TranscriptionFrame text for memory saving."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            _voice_memory_state["last_user_text"] = frame.text.strip()
            print(f"📝 User: {_voice_memory_state['last_user_text'][:50]}...")
        await self.push_frame(frame, direction)


# ======================================================================
# Voice Memory Inject Processor — injects retrieved memory into user turn
# ======================================================================
class VoiceMemoryInjectProcessor(FrameProcessor):
    def __init__(self, context: OpenAILLMContext):  # ← context pass karo
        super().__init__()
        self._context = context
        self._last_text = ""
        self._last_ts = 0.0

    def _inject_memory_as_system(self, user_text: str):
        try:
            from memory_search import smart_search_for_context
            memory_context = smart_search_for_context(user_text, limit=12, show_ui=True)
            if memory_context:
                msgs = list(self._context.get_messages())
                # Pehli dynamic memory hato, naya daalo
                msgs = [m for m in msgs
                        if not m.get("content","").startswith("RELEVANT MEMORY")]
                msgs.append({"role": "system", "content": memory_context})
                self._context.set_messages(msgs)
                
        except Exception as e:
            logger.warning("Voice memory inject failed: %s", e)
    def _is_duplicate(self, text: str) -> bool:
        import time
        now = time.time()

        if text == self._last_text and (now - self._last_ts) < 2:
            return True

        self._last_text = text
        self._last_ts = now
        return False   # ← same level pe lao
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text and getattr(frame, "finalized", True) and not self._is_duplicate(text):
                self._inject_memory_as_system(text)
                # ✅ frame.text TOUCH MAT KARO — original rehne do
        elif isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if text and not self._is_duplicate(text):
                self._inject_memory_as_system(text)
        await self.push_frame(frame, direction)

# ======================================================================
# LLM Output Sanitizer — strips meta/instruction leakage from response
# ======================================================================
class LLMOutputSanitizerProcessor(FrameProcessor):
    """Filters meta-commentary so spoken output stays natural."""

    META_PATTERNS = [
        r"\busing the information provided\b",
        r"\bthis approach ensures\b",
        r"\bin line with the rules\b",
        r"\bwithout breaking any constraints\b",
        r"\bi provide an appropriate response\b",
        r"\brule-breaker\b",
        r"\bwhile simultaneously providing valuable assistance\b",
    ]

    def __init__(self):
        super().__init__()
        self._compiled = [re.compile(p, flags=re.IGNORECASE) for p in self.META_PATTERNS]

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""

        out = text.strip()
        out = re.sub(r'^\s*Emily\s*:\s*["\']?', "", out, flags=re.IGNORECASE)

        if any(p.search(out) for p in self._compiled):
            return ""

        return out

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            cleaned = self._sanitize_text(frame.text)
            if not cleaned:
                return
            
            # Smart spacing fix
            text = cleaned
            
            # Fix contractions: I ' m → I'm
            text = re.sub(r"\s+'\s*", "'", text)
            
            # Fix punctuation: Hello , → Hello,
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            
            # Fix broken names: H iren → Hiren
            text = re.sub(r"\b([A-Z])\s+([a-z]{2,})", r"\1\2", text)
            
            # Normalize spaces
            text = re.sub(r"\s+", " ", text).strip()
            
            frame.text = text + " "
            
        await self.push_frame(frame, direction)


# ======================================================================
# Context Window Trim Processor — bounded in-session chat history
# ======================================================================
class ContextWindowTrimProcessor(FrameProcessor):
    """Keeps OpenAI context bounded so full session history doesn't bloat prompts."""

    def __init__(self, context: OpenAILLMContext, base_messages: int = 1, max_turn_messages: int = 4):
        super().__init__()
        self._context = context
        self._base_messages = max(1, int(base_messages))
        self._max_turn_messages = max(2, int(max_turn_messages))

    def _trim_context(self):
        try:
            msgs = list(self._context.get_messages())
            if len(msgs) <= self._base_messages + self._max_turn_messages:
                return

            base = msgs[: self._base_messages]
            tail = msgs[-self._max_turn_messages :]
            self._context.set_messages(base + tail)
        except Exception as e:
            logger.warning("Context trim failed: %s", e)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import LLMFullResponseEndFrame

        if isinstance(frame, LLMFullResponseEndFrame):
            self._trim_context()

        await self.push_frame(frame, direction)


# ======================================================================
# Memory Frame Processor — saves voice conversations to memories
# ======================================================================
class MemoryFrameProcessor(FrameProcessor):
    """
    Sits AFTER LLM in the pipeline.
    Gets user text from shared state (set by VoiceUserCaptureProcessor).
    Collects Emily's reply and saves the conversation pair to memory.
    """

    def __init__(self):
        super().__init__()
        self._emily_buffer = ""
        self._collecting = False
        
    def _clean_reply(self, reply: str) -> str:
       reply = re.sub(r'^\s*Emily\s*:\s*["\']?', "", reply, flags=re.IGNORECASE)
       reply = reply.strip('"\'').strip()
       return reply
      
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import (
            LLMFullResponseStartFrame,
            LLMFullResponseEndFrame,
        )

        # ── LLM starts streaming — begin collecting reply tokens ──
        if isinstance(frame, LLMFullResponseStartFrame):
            self._emily_buffer = ""
            self._collecting = True

        # ── Collect each LLM text token ──
        elif isinstance(frame, TextFrame) and self._collecting:
            self._emily_buffer += " " + frame.text

        # ── LLM done — save the full user+assistant pair to memory ──
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._collecting = False
            reply = self._clean_reply(self._emily_buffer.strip())
            user_text = _voice_memory_state.get("last_user_text", "").strip()
            
            if user_text and reply:
                try:
                    from memory_store import add_memory
                    print(f"💾 Saving memory: {user_text[:30]}...")
                    mem_id = add_memory(
                        content=f"User: {user_text}\nEmily: {reply}",
                        tags=["conversation", "voice"],
                        metadata={"mode": "voice_pipeline"},
                    )
                    print(f"✅ Memory saved: {mem_id}")
                    logger.info("🧠 Voice memory saved: %s", user_text[:50])
                except Exception as e:
                    print(f"❌ Memory save FAILED: {e}")
                    logger.warning("Voice memory save failed: %s", e)
            
            self._emily_buffer = ""

        await self.push_frame(frame, direction)


# ======================================================================
# Ollama model picker
# ======================================================================
def _pick_ollama_model() -> str:
    client = _ollama.Client(host=OLLAMA_BASE_URL)
    try:
        models = [m.model for m in client.list().models]
    except Exception as e:
        logger.error("Cannot list Ollama models: %s", e)
        print(f"{Fore.RED}❌  Ollama is not reachable. Start it with: ollama serve{Style.RESET_ALL}")
        sys.exit(1)

    if not models:
        print(f"{Fore.RED}❌  No models found. Run: ollama pull {DEFAULT_OLLAMA_MODEL}{Style.RESET_ALL}")
        sys.exit(1)

    if OLLAMA_MODEL and OLLAMA_MODEL in models:
        return OLLAMA_MODEL

    print(f"\n  Available Ollama models:")
    default_idx = 0
    for i, m in enumerate(models):
        tag = ""
        if m == DEFAULT_OLLAMA_MODEL:
            tag = f" {Fore.YELLOW}(recommended){Style.RESET_ALL}"
            default_idx = i
        print(f"    {i + 1}. {Fore.CYAN}{m}{Style.RESET_ALL}{tag}")

    try:
        choice = input(f"\n  Select model [{default_idx + 1}]: ").strip()
        idx    = int(choice) - 1 if choice else default_idx
        if not (0 <= idx < len(models)):
            idx = default_idx
    except (ValueError, EOFError):
        idx = default_idx

    selected = models[idx]
    print(f"\n  🧠  Selected: {Fore.CYAN}{selected}{Style.RESET_ALL}\n")
    return selected


# ======================================================================
# Voice Pipeline
# ======================================================================
async def run_voice_pipeline():
    selected_model = _pick_ollama_model()

    logger.info("=" * 50)
    logger.info("LLM MODEL SELECTED: %s", selected_model)
    logger.info("Ollama URL: %s/v1", OLLAMA_BASE_URL)
    logger.info("=" * 50)

    print(f"\n{Fore.MAGENTA}🤖 Voice Pipeline: {Fore.CYAN}{selected_model}{Style.RESET_ALL}\n")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=22050,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),           # ← wapas daalo
            vad_audio_passthrough=True,  # Piper TTS output rate
        )
    )

    stt = WhisperSTTService(
        model=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        language=None,
    )

    ollama_client   = _ollama.Client(host=OLLAMA_BASE_URL)
    image_processor = ImageRequestProcessor(ollama_client, selected_model)
    gesture_processor = GestureRequestProcessor()
    app_processor = AppRequestProcessor()
    app_processor._selected_model = selected_model
    music_processor = MusicRequestProcessor()
    folder_browser = FolderBrowserProcessor()

    llm = OLLamaLLMService(
        model=selected_model,
        base_url=f"{OLLAMA_BASE_URL}/v1",
    )

    tts = PiperTTSService(voice_id="en_US-amy-medium")

    # Build bounded startup context: system prompt + recent memory summary (8).
    startup_memory = ""
    try:
        from memory_search import format_memories_for_context

        startup_memory = format_memories_for_context(limit=8)
    except Exception as e:
        logger.warning("Startup memory preload failed: %s", e)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if startup_memory:
        messages.append({"role": "system", "content": startup_memory})

    # Preload custom commands for instant voice matching
    try:
        from custom_commands import load_commands
        custom_cmds = load_commands()
        if custom_cmds:
            print(f"🎯  Custom commands: {len(custom_cmds)} loaded for voice pipeline.")
    except Exception as e:
        logger.warning("Could not preload custom commands: %s", e)

    context            = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    memory_capture_processor = VoiceUserCaptureProcessor()
    memory_inject_processor = VoiceMemoryInjectProcessor(context)
    output_sanitizer = LLMOutputSanitizerProcessor()
    memory_processor = MemoryFrameProcessor()
    context_trimmer = ContextWindowTrimProcessor(
        context=context,
        base_messages=len(messages),
        max_turn_messages=4,
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        image_processor,
        gesture_processor,
        music_processor,            # Music control interceptor
        folder_browser,             # Folder browser interceptor
        app_processor,              # App open/close + intent engine interceptor
        memory_capture_processor,   # Captures latest user text for memory save
        memory_inject_processor,    # Injects retrieved memory context before LLM
        context_aggregator.user(),
        llm,
        output_sanitizer,           # Removes instruction/meta leakage from LLM output
        memory_processor,           # Saves exchange to LanceDB after LLM reply
        tts,
        transport.output(),
        context_aggregator.assistant(),
        context_trimmer,            # Keep only bounded recent turn history in context
    ])



    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    loop       = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        print(f"\n{Fore.YELLOW}👋  Shutting down. Goodbye boss!{Style.RESET_ALL}")
        stop_event.set()
        task.queue_frame(None)

    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
    except NotImplementedError:
        pass

    print(f"{Fore.GREEN}🎤  Pipeline ready — speak into your microphone!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}    Press Ctrl+C to stop{Style.RESET_ALL}\n")

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋  Shutting down. Goodbye boss!{Style.RESET_ALL}")
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        print(f"\n{Fore.RED}❌  Pipeline error: {e}{Style.RESET_ALL}")


# ======================================================================
# Text Mode
# ======================================================================
def run_text_mode():
    from llm_handler import LLMHandler
    from text_to_speech import TextToSpeech
    from rag_engine import RAGEngine

    llm = LLMHandler()
    tts = TextToSpeech()
    rag = RAGEngine()

    print(f"{Fore.CYAN}💬  TEXT MODE  –  type your messages")
    print(f"{Fore.YELLOW}    Commands:  quit | reset | voice | switch model | list models")
    print(f"{Fore.YELLOW}              insert info | print info | delete info")
    print(f"{Fore.YELLOW}              gesture volume | gesture brightness | gesture hand | stop gesture")
    print(f"{Fore.YELLOW}              show memories | search memory <query> | remember this <text>")
    print(f"{Fore.YELLOW}              open <app> | close <app> | list apps")
    print(f"{Fore.YELLOW}              play | pause | next | previous | volume up | volume down | mute | now playing")
    print(f"{Fore.YELLOW}              browse music | browse movies | play 1 | open 2 | go back\n")
    print(f"🤖  Emily: Text mode active, boss. Type away.\n")

    awaiting_delete = False

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            if not user_input.strip():
                continue

            lower = user_input.lower().strip()

            if lower in ("quit", "exit"):
                break
            if lower == "voice":
                asyncio.run(run_voice_pipeline())
                return
            if lower == "reset":
                llm.reset_conversation()
                print("🔄  Conversation reset.\n")
                continue
            if lower.startswith("switch model"):
                llm.switch_model()
                continue
            if lower.startswith("list model"):
                models = llm.list_models()
                for i, m in enumerate(models):
                    active = " ← active" if m == llm.model else ""
                    print(f"  {i+1}. {m}{active}")
                continue

            if awaiting_delete:
                awaiting_delete = False
                if "yes" in lower or "confirm" in lower:
                    rag.delete_info()
                    print("🤖  Emily: Vault wiped clean.\n")
                else:
                    print("🤖  Emily: Keeping everything.\n")
                continue

            if lower.startswith("insert info"):
                payload = user_input[len("insert info"):].strip()
                if payload:
                    rag.insert_info(payload)
                    print("🤖  Emily: Saved to vault.\n")
                continue
            if lower.startswith("print info"):
                content = rag.print_info()
                print(f"\n{Fore.CYAN}── Vault ──\n{content}\n{'─'*30}\n")
                continue
            if lower.startswith("delete info"):
                print("🤖  Emily: Delete everything? Say yes to confirm.")
                awaiting_delete = True
                continue

            # ── Gesture commands ──
            if lower.startswith("gesture volume"):
                from gesture_control import start_gesture
                print(f"🤖  Emily: {start_gesture('volume')}")
                continue
            if lower.startswith("gesture brightness"):
                from gesture_control import start_gesture
                print(f"🤖  Emily: {start_gesture('brightness')}")
                continue
            if lower.startswith("gesture hand") or lower == "gesture":
                from gesture_control import start_gesture
                print(f"🤖  Emily: {start_gesture('general')}")
                continue
            if lower.startswith("stop gesture"):
                from gesture_control import stop_gesture
                print(f"🤖  Emily: {stop_gesture()}")
                continue

            # ── Music commands ──
            if lower in ("play music", "play", "resume music", "resume"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.play()}")
                continue
            if lower in ("pause music", "pause", "stop music"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.pause()}")
                continue
            if lower in ("next song", "next track", "skip", "next"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.next_track()}")
                continue
            if lower in ("previous song", "previous track", "previous", "back"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.previous_track()}")
                continue
            if lower in ("volume up", "louder"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.volume_up()}")
                continue
            if lower in ("volume down", "quieter"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.volume_down()}")
                continue
            if lower in ("mute", "unmute"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.mute()}")
                continue
            if lower in ("what's playing", "current song", "now playing"):
                ctrl = get_music_controller()
                print(f"🤖  Emily: {ctrl.get_status()}")
                continue

            # ── Folder Browser commands ──
            browser_state = get_browser_state()
            
            # Handle selection when browser is active
            if browser_state["active"]:
                is_sel, num = is_browser_selection_command(user_input)
                if is_sel:
                    opener = get_file_opener()
                    result = asyncio.get_event_loop().run_until_complete(opener.select_item(num))
                    print(f"🤖  Emily: {result}")
                    continue
                
                nav_cmd = is_browser_navigation_command(user_input)
                if nav_cmd == "back":
                    opener = get_file_opener()
                    result = asyncio.get_event_loop().run_until_complete(opener.go_back())
                    print(f"🤖  Emily: {result}")
                    continue
                elif nav_cmd == "close":
                    opener = get_file_opener()
                    result = opener.close_browser()
                    print(f"🤖  Emily: {result}")
                    continue
            
            # Handle browse folder requests
            browse_keywords = ["browse", "show", "list", "open music", "open movies",
                              "open videos", "open pictures", "open documents", "open downloads"]
            folder_keywords = ["music", "movies", "videos", "pictures", "photos", "documents", "downloads", "songs", "gaane"]
            
            is_browse = any(kw in lower for kw in browse_keywords)
            has_folder = any(fk in lower for fk in folder_keywords)
            
            if is_browse and has_folder:
                opener = get_file_opener()
                # Extract folder name
                for fk in folder_keywords:
                    if fk in lower:
                        result, items = asyncio.get_event_loop().run_until_complete(opener.browse_folder(fk))
                        print(f"🤖  Emily: {result}")
                        # browse_folder already prints the items list, no need to duplicate
                        break
                continue

            # ── Memory commands ──
            if lower.startswith("show memor") or lower.startswith("list memor"):
                from memory_search import search_recent, count_memories
                total = count_memories()
                if total == 0:
                    print("🤖  Emily: No memories yet boss. Keep chatting!")
                else:
                    recent = search_recent(5)
                    print(f"🤖  Emily: {total} memories. Recent:")
                    for m in recent:
                        preview = m['content'][:80].replace('\n', ' ')
                        print(f"    {m['id']}: {preview}...")
                continue
            if lower.startswith("search memor"):
                query = user_input[len("search memory"):].strip()
                if query:
                    from memory_search import search_by_content
                    results = search_by_content(query)
                    if results:
                        print(f"🤖  Emily: Found {len(results)} memories:")
                        for m in results[:5]:
                            preview = m['content'][:80].replace('\n', ' ')
                            print(f"    {m['id']}: {preview}...")
                    else:
                        print(f"🤖  Emily: No memories matching '{query}'.")
                else:
                    print("🤖  Emily: Search for what? e.g. search memory python")
                continue
            if lower.startswith("remember this"):
                content = user_input[len("remember this"):].strip()
                if content:
                    from memory_store import add_memory
                    mem_id = add_memory(content, tags=["manual"])
                    print(f"🤖  Emily: Saved! ID: {mem_id}")
                else:
                    print("🤖  Emily: Remember what? e.g. remember this meeting at 5pm")
                continue

            # RAG context from vault
            context = rag.get_relevant_context(user_input)
            
            # Memory search - semantic search through past conversations (dynamic top-12)
            from memory_search import smart_search_for_context
            memory_context = smart_search_for_context(user_input, limit=12, show_ui=True)
            
            # Combine RAG and memory contexts
            if memory_context:
                context = f"{context}\n\n{memory_context}" if context else memory_context

            def on_sentence(sentence):
                tts.speak_async(sentence)

            reply = llm.chat(user_input, rag_context=context, sentence_callback=on_sentence)
            tts.wait()
            print()

        except KeyboardInterrupt:
            break

    print(f"\n{Fore.YELLOW}👋  Goodbye boss!{Style.RESET_ALL}")


# ======================================================================
# Entry point
# ======================================================================
def main():
    _start_graph_ui_server_if_enabled()

    print(f"{Fore.CYAN}  Select Mode:")
    print(f"{Fore.YELLOW}    1.  🎤  Voice Pipeline   (Pipecat real-time streaming)")
    print(f"{Fore.YELLOW}    2.  💬  Text Mode        (keyboard, for testing)")
    print()

    choice = input(f"{Fore.GREEN}  ➜  Enter choice (1/2): {Style.RESET_ALL}").strip()

    if choice == "2":
        run_text_mode()
    else:
        asyncio.run(run_voice_pipeline())


if __name__ == "__main__":
    main()
