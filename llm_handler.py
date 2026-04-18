import re
import os
import sys
from image_gen import generate_image
from gesture_control import start_gesture, stop_gesture
from memory_store import add_memory, get_all_memories
from memory_search import search_by_content, search_recent, count_memories
from commands import CommandExecutor
import json
import subprocess
import logging
import ollama
from colorama import Fore, Style
from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL, MAX_CONVERSATION_HISTORY,
    TEMPERATURE, MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Shared CommandExecutor instance
_command_executor = None

def get_command_executor():
    """Get or create the shared CommandExecutor instance."""
    global _command_executor
    if _command_executor is None:
        _command_executor = CommandExecutor()
    return _command_executor

# ── Model routing rules ───────────────────────────────────────────────
# Maps task categories to preferred model names (checked in order)
MODEL_ROUTES = {
    "coding": ["qwen3-coder:30b", "qwen3-coder"],
    "creative": ["dolphin:8b", "dolphin"],
    "general": ["gpt-oss:20b", "gpt-oss"],
}

# Keywords that trigger each route
ROUTE_KEYWORDS = {
    "coding": [
        "code", "program", "function", "debug", "script", "python",
        "javascript", "html", "css", "api", "class", "algorithm",
        "compile", "syntax", "error", "bug", "fix the", "write a function",
        "implement", "refactor", "deploy", "database", "sql",
    ],
    "creative": [
        "story", "poem", "joke", "sing", "song", "creative",
        "imagine", "fiction", "fairy tale", "horror", "adventure",
        "once upon", "tell me a", "write a story", "make up",
        "funny", "entertainment", "roleplay", "character",
    ],
}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _check_claude_code() -> bool:
    """Check if Claude Code CLI is installed and available."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info("Claude Code CLI found: %s", version)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


# ── Shared CommandExecutor singleton ────────────────────────────────
_cmd_executor = None

def get_command_executor():
    """Return a shared CommandExecutor instance (lazy init)."""
    global _cmd_executor
    if _cmd_executor is None:
        from commands import CommandExecutor
        _cmd_executor = CommandExecutor()
    return _cmd_executor


class LLMHandler:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = None
        self.default_model = None
        self.available_models: list[str] = []
        self.conversation_history: list[dict] = []
        self.saved_message_count = 0  # To track offset for saving new memories
        self.use_claude_code = False

        # System prompt is always first
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

        # Baseline starts clean; dynamic memory retrieval is injected per query.
        self.saved_message_count = len(self.conversation_history)

        # Preload custom commands (for instant matching on first message)
        try:
            from custom_commands import load_commands
            load_commands()
        except Exception as e:
            logger.warning("Could not preload custom commands: %s", e)

        # Discover models
        self._discover_models()

        # Pick default model
        if OLLAMA_MODEL:
            self.model = OLLAMA_MODEL
            print(f"✅  LLM model (from config): {self.model}")
        else:
            self.model = self._pick_model()
        self.default_model = self.model

        # Check if Claude Code is available
        self.use_claude_code = _check_claude_code()
        if self.use_claude_code:
            print(f"{Fore.GREEN}✅  Claude Code agent layer detected — full agentic mode enabled!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}    Model: {self.model}  →  via Claude Code + Ollama (offline){Style.RESET_ALL}")
            logger.info("Claude Code agent mode enabled with model: %s", self.model)
        else:
            print(f"{Fore.YELLOW}⚠️  Claude Code not found — using direct Ollama SDK{Style.RESET_ALL}")
            print(f"    To enable agentic mode: npm install -g @anthropic-ai/claude-code")
            logger.info("Direct Ollama mode with model: %s", self.model)

        # Print model routing info
        self._print_route_info()

    # ------------------------------------------------------------------
    # Save new memories using offset pattern
    # ------------------------------------------------------------------
    def save_new_memories(self, mode: str = "text"):
        """
        Saves only new messages that appeared since the last check.
        Pattern borrowed from memory_loop.py memory extractor.
        """
        if len(self.conversation_history) > self.saved_message_count:
            # Look for recent User -> Assistant pairs among the new messages
            new_msgs = self.conversation_history[self.saved_message_count:]
            
            # Find pairs
            for i in range(len(new_msgs) - 1):
                if new_msgs[i]["role"] == "user" and new_msgs[i+1]["role"] == "assistant":
                    user_text = new_msgs[i]["content"]
                    emily_text = new_msgs[i+1]["content"]
                    
                    try:
                        from memory_store import add_memory
                        print(f"💾 Saving text memory: {user_text[:30]}...")
                        mem_id = add_memory(
                            content=f"User: {user_text}\nEmily: {emily_text}",
                            tags=["conversation", mode],
                            metadata={"model": self.model}
                        )
                        print(f"✅ Text memory saved: {mem_id}")
                    except Exception as e:
                        print(f"❌ Text memory save FAILED: {e}")
                        logger.warning("Memory save failed: %s", e)

            # Important: update offset so we don't double-save
            self.saved_message_count = len(self.conversation_history)

    # ------------------------------------------------------------------
    # Intelligent model routing
    # ------------------------------------------------------------------
    def _route_model(self, user_message: str) -> str:
        """Pick the best model for this prompt based on keyword matching."""
        lower = user_message.lower()

        for category, keywords in ROUTE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                # Try each preferred model name for this category
                for preferred in MODEL_ROUTES[category]:
                    for available in self.available_models:
                        if preferred in available:
                            if available != self.model:
                                logger.info("Model routed: %s → %s (category: %s)",
                                            self.model, available, category)
                                print(f"  🔀  Routing to {Fore.CYAN}{available}{Style.RESET_ALL} ({category})")
                            return available
                break  # Matched a category but model not available

        # Fallback: default model
        return self.default_model

    def _print_route_info(self):
        """Show which models are available for routing."""
        print(f"\n  📋  Model Router:")
        for category, preferred_list in MODEL_ROUTES.items():
            matched = None
            for preferred in preferred_list:
                for available in self.available_models:
                    if preferred in available:
                        matched = available
                        break
                if matched:
                    break
            status = f"{Fore.GREEN}{matched}{Style.RESET_ALL}" if matched else f"{Fore.RED}not installed{Style.RESET_ALL}"
            print(f"    {category:10s} → {status}")
        print()

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------
    def _discover_models(self):
        """Fetch all models installed in Ollama."""
        try:
            result = self.client.list()
            self.available_models = [m.model for m in result.models]
            print(f"✅  Connected to Ollama  →  {len(self.available_models)} model(s) found")
            logger.info("Ollama connected: %d models", len(self.available_models))
        except Exception as e:
            print(f"❌  Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
            print("    Make sure Ollama is running:  ollama serve")
            logger.error("Ollama connection failed: %s", e)
            self.available_models = []

    def _pick_model(self) -> str:
        """Interactive model selector."""
        if not self.available_models:
            print(f"{Fore.RED}  No models found! Install one with: ollama pull qwen3-coder:30b")
            return DEFAULT_OLLAMA_MODEL

        print(f"\n{Fore.CYAN}  ╔══════════════════════════════════════╗")
        print(f"  ║   SELECT OLLAMA MODEL                ║")
        print(f"  ╚══════════════════════════════════════╝{Style.RESET_ALL}\n")

        default_idx = 0
        for i, name in enumerate(self.available_models):
            if DEFAULT_OLLAMA_MODEL and DEFAULT_OLLAMA_MODEL in name:
                default_idx = i

        for i, name in enumerate(self.available_models):
            marker = f" {Fore.GREEN}← default{Style.RESET_ALL}" if i == default_idx else ""
            print(f"  {Fore.YELLOW}{i + 1}.{Style.RESET_ALL}  {name}{marker}")

        print()
        choice = input(
            f"  {Fore.GREEN}➜ Pick model (1-{len(self.available_models)}) "
            f"[Enter = {default_idx + 1}]: {Style.RESET_ALL}"
        ).strip()

        if not choice:
            idx = default_idx
        else:
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(self.available_models):
                    idx = default_idx
            except ValueError:
                idx = default_idx

        selected = self.available_models[idx]
        print(f"\n  🧠  Selected: {Fore.CYAN}{selected}{Style.RESET_ALL}\n")
        return selected

    # ------------------------------------------------------------------
    # Switch model at runtime
    # ------------------------------------------------------------------
    def switch_model(self) -> str:
        """Let the user pick a different model."""
        self._discover_models()
        old = self.model
        self.model = self._pick_model()
        if self.model != old:
            self.reset_conversation()
            print(f"🔄  Switched from {old} → {self.model}")
        return self.model

    def list_models(self) -> list[str]:
        self._discover_models()
        return self.available_models

    def _is_image_request(self, text: str) -> bool:
        keywords = [
           "generate image", "create image", "make image",
           "draw", "image of", "picture of", "show me a picture",
           "image banao", "tasveer banao", "photo banao",
           "ek image", "generate a photo", "create a photo",
           "paint", "sketch", "visualize",
        ]
        return any(kw in text.lower() for kw in keywords)

    def _is_gesture_request(self, text: str) -> tuple[bool, str]:
        """Check if the user wants gesture control. Returns (is_gesture, mode)."""
        lower = text.lower()
        
        # Stop gesture - check FIRST (highest priority)
        stop_kw = ["stop gesture", "gesture stop", "gesture band karo", "gesture off",
                   "camera off", "camera band karo", "hand tracking off", "stop hand",
                   "close gesture", "close hand gesture", "exit gesture", "gesture close",
                   "gesture exit", "hand gesture off", "hand gesture band",
                   "close camera", "end gesture", "gesture end"]
        if any(kw in lower for kw in stop_kw):
            return True, "stop"
        
        # Brightness - check for "bright" with gesture context
        if "bright" in lower and ("gesture" in lower or "hand" in lower or "control" in lower):
            return True, "brightness"
        
        # Volume - check for "volume" with gesture context
        if "volume" in lower and ("gesture" in lower or "hand" in lower or "control" in lower):
            return True, "volume"
        
        # General gesture / hand detection - only if no specific mode
        gen_kw = ["gesture mode", "gesture control", "hand detection",
                  "hand tracking", "camera control", "start gesture",
                  "haath dikhao", "gesture chalu karo"]
        if any(kw in lower for kw in gen_kw):
            return True, "general"
        
        # "hand gesture" alone (without brightness/volume) = general
        if "hand gesture" in lower and "bright" not in lower and "volume" not in lower:
            return True, "general"
        
        return False, ""

    def _is_memory_request(self, text: str) -> tuple[bool, str]:
        """Check if user wants memory operations. Returns (is_memory, action)."""
        lower = text.lower()
        # Show/list memories
        show_kw = ["show memor", "list memor", "my memories", "all memories",
                   "memory dikha", "memories dikha", "kitni memory"]
        if any(kw in lower for kw in show_kw):
            return True, "show"
        # Search memories
        search_kw = ["search memor", "find memor", "memory search",
                     "memory dhundho", "yaad hai"]
        if any(kw in lower for kw in search_kw):
            return True, "search"
        # Save memory explicitly
        save_kw = ["save memory", "remember this", "yaad rakh",
                   "memory save", "store memory", "save this"]
        if any(kw in lower for kw in save_kw):
            return True, "save"
        # Destructive memory/vault actions
        clear_kw = ["clear memory", "wipe memory", "delete memory", "delete all memories",
                    "clear vault", "delete vault", "wipe vault", "delete info", "clear info"]
        if any(kw in lower for kw in clear_kw):
            return True, "clear_confirm"
        return False, ""

    # Compound command separators
    COMPOUND_SEPARATORS = [
        " and then ", " then ", " and ", " aur phir ", " phir ", " aur ", 
        ", then ", ", and ", ", ", " uske baad ",
    ]
    
    # Keywords that indicate content should be GENERATED, not typed literally
    GENERATION_TRIGGERS = [
        "random", "generate", "create", "make", "banao", "bana do",
        "paragraph", "story", "poem", "essay", "letter", "email",
        "code", "function", "script", "program",
        "words", "lines", "sentences",
        "about", "on topic", "regarding", "ke baare me",
        "something", "kuch", "anything",
    ]
    
    def _needs_generation(self, text: str) -> bool:
        """Check if the text needs LLM generation (not literal typing)."""
        lower = text.lower()
        return any(trigger in lower for trigger in self.GENERATION_TRIGGERS)
    
    def _generate_content(self, prompt: str) -> str:
        """Generate content using LLM based on user's request."""
        try:
            # Create a focused prompt for content generation
            system_prompt = """You are a content generator. Generate ONLY the requested content.
Do NOT add explanations, introductions, or "Here's..." phrases.
Do NOT use markdown formatting.
Output ONLY the raw content that should be typed."""
            
            response = self.client.chat(
                model=self.model,
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
            logger.warning("Content generation failed; typing original prompt instead.")
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
    
    def _is_app_request(self, text: str) -> tuple[bool, str, str]:
        """
        Check if user wants to open/close app, type, open website, or search.
        Returns: (is_command, action, target)
        
        Supported actions:
        - "open", "close", "list", "type", "website", "search"
        """
        lower = text.lower().strip()
        
        # List running apps
        list_kw = ["list apps", "running apps", "open apps", "kya chal raha",
                   "konsi app", "kitni apps", "show apps", "active windows"]
        if any(kw in lower for kw in list_kw):
            return True, "list", ""
        
        # Search keywords
        search_kw = ["search ", "google karo ", "google search ", "search for ",
                     "google pe dhundo ", "search karo ", "dhundo "]
        for kw in search_kw:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    for suffix in ["karo", "kar", "please", "boss"]:
                        query = query.replace(suffix, "").strip()
                    if query:
                        return True, "search", query
        
        # Type text keywords
        type_kw = ["type ", "likho ", "likh do ", "write ", "likho yeh ",
                   "type this ", "likho ki ", "likh ", "type the text "]
        for kw in type_kw:
            if kw in lower:
                parts = lower.split(kw, 1)
                if len(parts) > 1:
                    text_to_type = parts[1].strip().strip('"\'')
                    if text_to_type:
                        # Check if this needs generation
                        if self._needs_generation(text_to_type):
                            return True, "generate_type", text_to_type
                        return True, "type", text_to_type
        
        # Close app keywords (check FIRST - more specific)
        close_kw = ["close ", "band karo", "band kar", "exit ", "quit ", 
                    "bund karo", "bund kar", "hatao", "shut down ",
                    "close the", "band kardo"]
        for kw in close_kw:
            if kw in lower:
                # Extract app name after the keyword
                if kw in lower:
                    parts = lower.split(kw, 1)
                    if len(parts) > 1:
                        app_name = parts[1].strip()
                        # Clean common suffixes
                        for suffix in ["karo", "kar", "do", "please", "boss"]:
                            app_name = app_name.replace(suffix, "").strip()
                        if app_name:
                            return True, "close", app_name
        
        # Known websites that should open directly
        known_sites = ["amazon", "flipkart", "youtube", "google", "facebook", "instagram",
                       "twitter", "linkedin", "reddit", "netflix", "spotify", "github",
                       "gmail", "whatsapp", "telegram", "wikipedia", "pinterest", "twitch",
                       "stackoverflow", "chatgpt", "claude", "notion", "hotstar", "myntra"]
        
        # Open app/website keywords
        open_kw = ["open ", "kholo", "khol do", "launch ", "start ", 
                   "chalu karo", "chalu kar", "run ", "open the"]
        for kw in open_kw:
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
                        for site in known_sites:
                            if site in target_clean or target_clean in site:
                                return True, "website", target
                        
                        # Check if it looks like a URL
                        if ".com" in target or ".in" in target or ".org" in target or "www." in target:
                            return True, "website", target
                        
                        # Otherwise treat as app
                        return True, "open", target
        
        return False, "", ""

    def _handle_app_request(self, action: str, target: str) -> str:
        """
        Handle app/type/website/search requests using CommandExecutor.
        """
        import time
        executor = get_command_executor()
        
        if action == "open":
            return executor.open_application(target)
        elif action == "close":
            return executor.close_application(target)
        elif action == "list":
            return executor.list_running_apps()
        elif action == "type":
            return executor.type_text(target)
        elif action == "generate_type":
            # Generate content using LLM, then type it
            print(f"✨ Generating content for: {target}")
            generated = self._generate_content(target)
            if generated and generated.strip():
                print(f"✨ Generated {len(generated)} chars, typing...")
                return executor.type_text(generated)
            print("⚠️ Generation unavailable, typing requested text directly.")
            return executor.type_text(target)
        elif action == "website":
            return executor.open_website(target)
        elif action == "search":
            return executor.search_web(target)
        else:
            return "Sorry boss, I didn't understand that command."
    
    def _handle_compound_command(self, text: str) -> tuple[bool, str]:
        """
        Handle compound commands like "open notepad and type hello".
        Returns: (was_compound, result_message)
        """
        import time
        commands = self._split_compound_command(text)
        
        if len(commands) == 1:
            # Single command - check if it's a valid app request
            is_app, action, target = self._is_app_request(commands[0])
            if is_app:
                result = self._handle_app_request(action, target)
                return True, result
            return False, ""
        
        # Multiple commands - execute sequentially
        results = []
        print(f"🔗 Compound command detected: {len(commands)} parts")
        
        for i, cmd in enumerate(commands):
            is_app, action, target = self._is_app_request(cmd)
            if is_app:
                print(f"   [{i+1}] {action}: {target}")
                result = self._handle_app_request(action, target)
                results.append(result)
                
                # Wait between commands (especially after open)
                if action == "open" and i < len(commands) - 1:
                    time.sleep(2.0)  # Wait for app to open
                elif i < len(commands) - 1:
                    time.sleep(0.5)  # Small delay between other commands
        
        if results:
            # Return summary
            if len(results) == 1:
                return True, results[0]
            else:
                return True, f"Done boss! Executed {len(results)} commands."
        
        return False, ""

    # ------------------------------------------------------------------
    # Chat — Claude Code Agent Mode
    # ------------------------------------------------------------------
    def _chat_claude_code(self, user_message: str, rag_context: str = "", sentence_callback=None) -> str:
        """
        Send message through Claude Code CLI → Ollama (offline).
        Claude Code acts as the agent layer with full tool capabilities.
        """
        if rag_context:
            full_prompt = (
                f"Context from knowledge base:\n{rag_context}\n\n"
                f"User: {user_message}"
            )
        else:
            full_prompt = user_message

        # Build environment with Ollama endpoint
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = OLLAMA_BASE_URL
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"

        # Build claude command
        cmd = [
            "claude",
            "-p",                       # Non-interactive (print mode)
            "--model", self.model,      # Use selected Ollama model
            "--system-prompt", SYSTEM_PROMPT,
            "--output-format", "text",  # Plain text output
            "--max-turns", "1",         # Single response for voice
            full_prompt,
        ]

        try:
            sys.stdout.write(f"🤖  Emily: ")
            sys.stdout.flush()

            # Run Claude Code and stream output
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=os.getcwd(),
                encoding="utf-8",
                errors="replace",
            )

            full_reply = ""
            sentence_buffer = ""
            # Read output character by character for streaming
            while True:
                char = proc.stdout.read(1)
                if not char:
                    break
                full_reply += char
                sys.stdout.write(char)
                sys.stdout.flush()
                
                if sentence_callback:
                    sentence_buffer += char
                    if any(sentence_buffer.endswith(p) for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"]):
                        clean_sentence = _strip_think_tags(sentence_buffer).strip()
                        if clean_sentence:
                            sentence_callback(clean_sentence)
                        sentence_buffer = ""

            proc.wait(timeout=60)
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Process remainder
            if sentence_callback and sentence_buffer.strip():
                clean_remainder = _strip_think_tags(sentence_buffer).strip()
                if clean_remainder:
                    sentence_callback(clean_remainder)

            if proc.returncode != 0:
                stderr = proc.stderr.read()
                if stderr:
                    logger.warning("Claude Code stderr: %s", stderr[:300])

            # Strip think tags
           # Strip think tags
            reply = _strip_think_tags(full_reply.strip())

            # If stripping <think> tags left nothing
            if not reply.strip():
                after_think = re.split(r"</think>", full_reply, flags=re.DOTALL)
                if len(after_think) > 1 and after_think[-1].strip():
                    reply = after_think[-1].strip()
                else:
                    reply = "Could you repeat that? I want to make sure I give you the right answer."
                logger.warning("Claude Code returned only <think> content, using fallback reply")

            # Store in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": reply,
            })
            self._trim_history()

            logger.info("Claude Code response: %d chars", len(reply))
            return reply

        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error("Claude Code timed out")
            return "Sorry boss, that took too long. Let me try again."
        except Exception as e:
            logger.error("Claude Code error: %s", e)
            # Fallback to direct Ollama
            print(f"\n⚠️  Claude Code error, falling back to direct Ollama...")
            return self._chat_ollama_direct(user_message, rag_context, sentence_callback)

    # ------------------------------------------------------------------
    # Chat — Direct Ollama Fallback
    # ------------------------------------------------------------------
    def _chat_ollama_direct(self, user_message: str, rag_context: str = "", sentence_callback=None) -> str:
        """Direct Ollama SDK call — fallback when Claude Code is unavailable."""
        if rag_context:
            augmented = (
                f"Use the following context to help answer the user's query. "
                f"Don't mention that you received context, just use it naturally.\n\n"
                f"--- CONTEXT ---\n{rag_context}\n--- END CONTEXT ---\n\n"
                f"User: {user_message}"
            )
        else:
            augmented = user_message

        self.conversation_history.append({
            "role": "user",
            "content": augmented,
        })

        try:
            # Route to best model for this prompt
            routed_model = self._route_model(user_message)

            # Stream response token-by-token (think=False disables CoT for qwen3 etc.)
            stream = self.client.chat(
                model=routed_model,
                messages=self.conversation_history,
                stream=True,
                think=False,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                },
            )

            full_reply = ""
            sentence_buffer = ""
            sys.stdout.write(f"🤖  Emily: ")
            sys.stdout.flush()

            for chunk in stream:
                token = chunk.message.content
                if token:
                    full_reply += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    
                    if sentence_callback:
                        sentence_buffer += token
                        if any(sentence_buffer.endswith(p) for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"]):
                            clean_sentence = _strip_think_tags(sentence_buffer).strip()
                            if clean_sentence:
                                sentence_callback(clean_sentence)
                            sentence_buffer = ""

            sys.stdout.write("\n")
            sys.stdout.flush()

            # Process remainder
            if sentence_callback and sentence_buffer.strip():
                clean_remainder = _strip_think_tags(sentence_buffer).strip()
                if clean_remainder:
                    sentence_callback(clean_remainder)

            reply = _strip_think_tags(full_reply)

            # If stripping <think> tags left nothing — model replied only in thinking
            # Try to use whatever remains, or give a safe fallback
            if not reply.strip():
                # Check if there's any text after the last </think>
                after_think = re.split(r"</think>", full_reply, flags=re.DOTALL)
                if len(after_think) > 1 and after_think[-1].strip():
                    reply = after_think[-1].strip()
                else:
                    reply = "Could you repeat that? I want to make sure I give you the right answer."
                logger.warning("LLM returned only <think> content, using fallback reply")

            self.conversation_history.append({
                "role": "assistant",
                "content": reply,
            })

            self._trim_history()
            logger.info("Ollama direct response: %d chars", len(reply))
            return reply

        except Exception as e:
            print(f"❌  LLM error: {e}")
            logger.error("Ollama direct error: %s", e)
            return "Sorry boss, my brain just glitched. Could you say that again?"

    # ------------------------------------------------------------------
    # Main chat entry point
    # ------------------------------------------------------------------
    def chat(self, user_message: str, rag_context: str = "", sentence_callback=None) -> str:

        def _respond(text: str) -> str:
            print(f"🤖  Emily: {text}")
            if sentence_callback:
                sentence_callback(text)
            return text

        # ══ Pending destructive confirmation (highest priority) ═══════════
        if getattr(self, "_pending_destructive_action", None):
            return _respond(self._handle_memory(user_message, "confirm_pending"))

        # ══ LAYER 1: Custom Command Match (highest priority, instant) ═══
        try:
            from custom_commands import match_command, execute_command, \
                                        parse_learn_intent, save_command, load_commands
            load_commands()

            # 1a. Detect "when I say X, do Y" — learn new command
            learn = parse_learn_intent(user_message)
            if learn:
                trigger  = learn["trigger"]
                actions  = learn["actions"]
                ok, msg  = save_command(trigger, actions)
                confirm  = (f"Got it boss! {msg} "
                            f"Say '{trigger}' anytime to run: {', '.join(actions)}.")
                return _respond(confirm)

            # 1b. Check if input matches a saved custom command
            matched = match_command(user_message)
            if matched:
                result = execute_command(matched, executor=get_command_executor())
                return _respond(result)

        except Exception as e:
            logger.warning("Custom command layer error: %s", e)

        # ══ LAYER 2: Intent Engine (system commands, skip LLM) ═══════════
        try:
            from intent_engine import detect_intent, \
                OPEN_APP, CLOSE_APP, SEARCH_WEB, TYPE_TEXT, PLAY_MUSIC, CONTROL_SYSTEM
            from commands import CommandExecutor

            intent = detect_intent(user_message)
            itype  = intent["type"]
            target = intent["target"]

            if itype == OPEN_APP and target:
                result = get_command_executor().open_application(target)
                return _respond(result)

            elif itype == CLOSE_APP and target:
                result = get_command_executor().close_application(target)
                return _respond(result)

            elif itype == SEARCH_WEB and target:
                import webbrowser
                query = target.replace(" ", "+")
                webbrowser.open(f"https://www.google.com/search?q={query}")
                return _respond(f"Done boss, searching for '{target}' in your browser!")

            elif itype == TYPE_TEXT and target:
                result = get_command_executor().type_text(target)
                return _respond(result)

            elif itype == PLAY_MUSIC:
                try:
                    from music_control import play_music
                    result = play_music(target or "")
                    return _respond(result)
                except Exception:
                    pass  # Fall through to LLM if music_control unavailable

        except Exception as e:
            logger.warning("Intent engine layer error: %s", e)

        # ══ Dynamic memory context (semantic top-k, per query) ═══════════
        try:
            from memory_search import smart_search_for_context
            dynamic_memory = smart_search_for_context(user_message, limit=12, show_ui=True)
            if dynamic_memory:
                rag_context = f"{rag_context}\n\n{dynamic_memory}" if rag_context else dynamic_memory
        except Exception as e:
            logger.warning("Dynamic memory injection failed: %s", e)

        # ── Gesture control interception ──
        is_gesture, gesture_mode = self._is_gesture_request(user_message)
        if is_gesture:
            if gesture_mode == "stop":
                result = stop_gesture()
            else:
                result = start_gesture(gesture_mode)
            return _respond(result)

        # ── App open/close interception (with compound command support) ──
        is_compound, compound_result = self._handle_compound_command(user_message)
        if is_compound:
            return _respond(compound_result)

        # ── Memory commands interception ──
        is_mem, mem_action = self._is_memory_request(user_message)
        if is_mem:
            return _respond(self._handle_memory(user_message, mem_action))

        # ── Image generation interception ──
        if self._is_image_request(user_message):
            _respond("Sure boss, generating your image right now. Please wait a moment.")
            return generate_image(
                prompt=user_message,
                llm_client=self.client,
                llm_model=self.model,
            )

        # ── LLM routing ──
        if self.use_claude_code:
            reply = self._chat_claude_code(user_message, rag_context, sentence_callback)
        else:
            reply = self._chat_ollama_direct(user_message, rag_context, sentence_callback)

        # ── Auto-save conversation to memory using offset pattern ──
        self.save_new_memories(mode="text")

        return reply

    # ------------------------------------------------------------------
    def _handle_memory(self, user_message: str, action: str) -> str:
        """Handle memory commands."""
        if not hasattr(self, "_pending_destructive_action"):
            self._pending_destructive_action = None

        # Handle confirmation reply if pending
        lower = user_message.lower().strip()
        if self._pending_destructive_action:
            if lower in ("yes", "confirm", "do it", "haan", "ha", "y", "yes boss"):
                pending = self._pending_destructive_action
                self._pending_destructive_action = None
                if pending == "delete_vault":
                    try:
                        from rag_engine import RAGEngine
                        RAGEngine().delete_info()
                        return "Vault cleared successfully, boss."
                    except Exception as e:
                        return f"Sorry boss, couldn't clear vault: {e}"
                return "No destructive action found, boss."
            if lower in ("no", "cancel", "stop", "n", "no boss"):
                self._pending_destructive_action = None
                return "Cancelled boss. Nothing was deleted."
            return "Please say 'yes' to confirm or 'no' to cancel."

        if action == "show":
            total = count_memories()
            if total == 0:
                return "No memories saved yet boss. Just keep chatting and I'll remember everything."
            # Use search_recent with UI feedback
            recent = search_recent(5, show_ui=True)
            lines = [f"I have {total} memories boss. Here are the latest:"]
            for m in recent:
                content_preview = m['content'][:80].replace('\n', ' ')
                lines.append(f"  {m['id']}: {content_preview}...")
            return "\n".join(lines)

        elif action == "search":
            # Extract search query from the message
            query = user_message.lower()
            for kw in ["search memory", "find memory", "memory search",
                       "memory dhundho", "yaad hai"]:
                query = query.replace(kw, "").strip()
            if not query:
                return "What should I search for boss? Say something like: search memory python"
            
            # Use smart search with UI feedback
            from memory_search import smart_search_for_context
            context = smart_search_for_context(query, limit=5, show_ui=True)
            
            if not context:
                # Fallback to keyword search
                results = search_by_content(query)
                if not results:
                    return f"No memories found matching '{query}' boss."
                lines = [f"Found {len(results)} memories for '{query}':"]
                for m in results[:5]:
                    content_preview = m['content'][:80].replace('\n', ' ')
                    lines.append(f"  {m['id']}: {content_preview}...")
                return "\n".join(lines)
            
            return f"Found relevant memories for '{query}'. Check the details above boss!"

        elif action == "save":
            # Extract content to save
            content = user_message
            for kw in ["save memory", "remember this", "yaad rakh",
                       "memory save", "store memory", "save this"]:
                content = content.lower().replace(kw, "").strip()
            if not content:
                return "What should I remember boss? Say something like: remember this meeting is at 5pm"
            mem_id = add_memory(content, tags=["manual", "user_saved"])
            return f"Saved to memory boss! ID: {mem_id}"

        elif action == "clear_confirm":
            # Only vault clear supported in current storage APIs
            self._pending_destructive_action = "delete_vault"
            return "This will delete all vault info. Say 'yes' to confirm or 'no' to cancel."

        return "I didn't understand that memory command boss."

    # ------------------------------------------------------------------
    def _trim_history(self):
        max_msgs = MAX_CONVERSATION_HISTORY * 2 + 1
        if len(self.conversation_history) > max_msgs:
            self.conversation_history = (
                [self.conversation_history[0]]
                + self.conversation_history[-MAX_CONVERSATION_HISTORY * 2:]
            )

    def reset_conversation(self):
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("🔄  Conversation history cleared.")


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = LLMHandler()
    print(f"\nUsing model: {llm.model}")
    print(f"Agent mode: {'Claude Code' if llm.use_claude_code else 'Direct Ollama'}")
    print(f"Available: {llm.available_models}")

    while True:
        msg = input("\nYou: ")
        if msg.lower() in ("exit", "quit"):
            break
        if msg.lower() == "switch":
            llm.switch_model()
            continue
        llm.chat(msg)
