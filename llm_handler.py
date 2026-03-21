"""
LLM Handler — Claude Code Agent + Ollama (100% Offline)
=======================================================
Uses Claude Code CLI as the agent layer to talk to local Ollama models.
This gives the assistant full agentic capabilities:
  • File editing, terminal execution, codebase understanding
  • Tool use / multi-step agent tasks
  • All running locally through Ollama — no internet needed

Setup:
  1. Install Claude Code:  npm install -g @anthropic-ai/claude-code
  2. Install Ollama:       https://ollama.com
  3. Pull a model:         ollama pull qwen3-coder:30b
  4. Run:                  python main.py

Fallback: If Claude Code CLI is not available, uses Ollama Python SDK directly.
"""

import re
import os
import sys
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


class LLMHandler:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = None
        self.default_model = None
        self.available_models: list[str] = []
        self.conversation_history: list[dict] = []
        self.use_claude_code = False

        # System prompt is always first
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

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
            sys.stdout.write(f"🤖  Jarvis: ")
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

            # Stream response token-by-token
            stream = self.client.chat(
                model=routed_model,
                messages=self.conversation_history,
                stream=True,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                },
            )

            full_reply = ""
            sentence_buffer = ""
            sys.stdout.write(f"🤖  Jarvis: ")
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
        """
        Send a message to the LLM.
        Routes through Claude Code agent if available, else direct Ollama.
        """
        if self.use_claude_code:
            return self._chat_claude_code(user_message, rag_context, sentence_callback)
        else:
            return self._chat_ollama_direct(user_message, rag_context, sentence_callback)

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
