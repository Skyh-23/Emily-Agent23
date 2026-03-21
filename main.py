"""
J.A.R.V.I.S  —  Local Voice Assistant (Pipecat Pipeline)
=========================================================
Fully offline, real-time streaming voice pipeline using Pipecat.

Pipeline:  Microphone → Silero VAD → Whisper STT → Ollama LLM → Piper TTS → Speaker

Usage:  python main.py
"""

import sys
import io
# Fix Windows encoding issues with UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from colorama import Fore, Style, init as colorama_init
colorama_init(strip=False)

BANNER = f"""
{Fore.CYAN}==============================================================
=                                                            =
=   [MIC]  J.A.R.V.I.S  —  Pipecat Voice Pipeline         =
=         100% Offline  ·  Real-Time Streaming             =
=         Whisper STT  ·  Ollama LLM  ·  Piper TTS        =
=                                                            =
=============================================================={Style.RESET_ALL}
"""
print(BANNER)
print(f"{Fore.YELLOW}[LOADING]  Loading AI Models and Modules... Please wait (10-30 seconds).{Style.RESET_ALL}\n")
sys.stdout.flush()

import asyncio
import signal
import os
import logging

# ── Pipecat core ──────────────────────────────────────────────────────
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner

# ── Transport ─────────────────────────────────────────────────────────
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

# ── VAD ───────────────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer

# ── Services (all local / offline) ────────────────────────────────────
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService

# ── Context aggregator ────────────────────────────────────────────────
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS,
    LOG_LEVEL, LOG_FILE,
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(name)-20s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ======================================================================
# Ollama model picker
# ======================================================================
def _pick_ollama_model() -> str:
    """Pick an Ollama model — honour config or let user choose."""
    import ollama as _ollama

    client = _ollama.Client(host=OLLAMA_BASE_URL)
    try:
        models = [m.model for m in client.list().models]
    except Exception as e:
        logger.error("Cannot list Ollama models: %s", e)
        print(f"{Fore.RED}❌  Ollama is not reachable at {OLLAMA_BASE_URL}")
        print(f"    Start it with:  ollama serve{Style.RESET_ALL}")
        sys.exit(1)

    if not models:
        print(f"{Fore.RED}❌  No models found. Pull one first:")
        print(f"    ollama pull {DEFAULT_OLLAMA_MODEL}{Style.RESET_ALL}")
        sys.exit(1)

    if OLLAMA_MODEL and OLLAMA_MODEL in models:
        return OLLAMA_MODEL

    # Interactive picker
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
        idx = int(choice) - 1 if choice else default_idx
        if not (0 <= idx < len(models)):
            idx = default_idx
    except (ValueError, EOFError):
        idx = default_idx

    selected = models[idx]
    print(f"\n  🧠  Selected: {Fore.CYAN}{selected}{Style.RESET_ALL}\n")
    return selected


# ======================================================================
# Pipecat Voice Mode (real-time streaming pipeline)
# ======================================================================
async def run_voice_pipeline():
    """Real-time streaming voice pipeline using Pipecat."""

    # ── Model picker — sirf BAAR ──────────────────────────────────
    selected_model = _pick_ollama_model()

    logger.info("=" * 50)
    logger.info("LLM MODEL SELECTED: %s", selected_model)
    logger.info("Ollama URL: %s/v1", OLLAMA_BASE_URL)
    logger.info("=" * 50)

    print(f"\n{Fore.MAGENTA}🤖 Voice Pipeline active with AI model: {Fore.CYAN}{selected_model}{Style.RESET_ALL}\n")

    # ── Transport ─────────────────────────────────────────────────
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    # ── STT (Whisper) ─────────────────────────────────────────────
    stt = WhisperSTTService(
        model=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        language=None,              # auto-detect — Hindi + English + Hinglish
    )

    # ── LLM (Ollama) ──────────────────────────────────────────────
    llm = OLLamaLLMService(
        model=selected_model,
        base_url=f"{OLLAMA_BASE_URL}/v1",
    )

    # ── TTS (Piper) ────────────────────────────────────────────────
    tts = PiperTTSService(
        voice_id="en_US-amy-medium",
    )

    # ── Context ───────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ── Pipeline ──────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Ctrl+C handling (Windows-safe) ────────────────────────────
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        print(f"\n{Fore.YELLOW}👋  Shutting down pipeline. Goodbye boss!{Style.RESET_ALL}")
        stop_event.set()
        task.queue_frame(None)

    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
    except NotImplementedError:
        pass

    # ── Run ───────────────────────────────────────────────────────
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
        
        
def run_text_mode():
    """Text-only fallback mode using llm_handler + streaming TTS."""
    from llm_handler import LLMHandler
    from text_to_speech import TextToSpeech
    from rag_engine import RAGEngine

    llm = LLMHandler()
    tts = TextToSpeech()
    rag = RAGEngine()

    print(f"{Fore.CYAN}💬  TEXT MODE  –  type your messages")
    print(f"{Fore.YELLOW}    Commands:  quit | reset | voice | switch model | list models")
    print(f"{Fore.YELLOW}              insert info | print info | delete info\n")
    print(f"🤖  Jarvis: Text mode active, boss. Type away.\n")

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

            # ── Delete confirm ────────────────────────────────────
            if awaiting_delete:
                awaiting_delete = False
                if "yes" in lower or "confirm" in lower:
                    rag.delete_info()
                    print("🤖  Jarvis: Vault wiped clean.\n")
                else:
                    print("🤖  Jarvis: Keeping everything.\n")
                continue

            # ── Voice commands (text) ─────────────────────────────
            if lower.startswith("insert info"):
                payload = user_input[len("insert info"):].strip()
                if payload:
                    rag.insert_info(payload)
                    print("🤖  Jarvis: Saved to vault.\n")
                continue
            if lower.startswith("print info"):
                content = rag.print_info()
                print(f"\n{Fore.CYAN}── Vault ──\n{content}\n{'─'*30}\n")
                continue
            if lower.startswith("delete info"):
                print("🤖  Jarvis: Delete everything? Say yes to confirm.")
                awaiting_delete = True
                continue

            # ── LLM chat with streaming TTS ───────────────────────
            context = rag.get_relevant_context(user_input)

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
