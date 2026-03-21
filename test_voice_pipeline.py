"""
Quick Voice Pipeline Test
Tests the complete end-to-end pipeline with real-time interaction
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
from config import (
    OLLAMA_BASE_URL, WHISPER_MODEL, WHISPER_DEVICE, 
    WHISPER_COMPUTE_TYPE, SYSTEM_PROMPT
)

print("[*] Starting Voice Pipeline Test...")
print(f"    Ollama Server: {OLLAMA_BASE_URL}")
print(f"    Whisper Model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
print()

async def test_pipeline():
    """Test the Pipecat voice pipeline with Ollama integration."""
    
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.pipeline.runner import PipelineRunner
    
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.piper.tts import PiperTTSService
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.processors.aggregators.sentence import SentenceAggregator
    from pipecat.frames.frames import TextFrame, Frame
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

    # ── VAD ───────────────────────────────────────────────────────
    try:
        vad = SileroVADAnalyzer()
        print("[✓] VAD (Silero) loaded")
    except Exception as e:
        print(f"[!] VAD loading failed: {e}")
        return

    # ── Transport ─────────────────────────────────────────────────
    try:
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=vad,
                vad_audio_passthrough=True,
            )
        )
        print("[✓] Audio Transport initialized")
    except Exception as e:
        print(f"[✗] Audio Transport failed: {e}")
        return

    # ── Services ──────────────────────────────────────────────────
    try:
        stt = WhisperSTTService(
            model=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[✓] Whisper STT initialized")
    except Exception as e:
        print(f"[✗] Whisper STT failed: {e}")
        return

    try:
        llm = OLLamaLLMService(
            model="dolphin-llama3:8b",  # Use a fast model for testing
            base_url=OLLAMA_BASE_URL,
        )
        print("[✓] Ollama LLM initialized")
    except Exception as e:
        print(f"[✗] Ollama LLM failed: {e}")
        return

    try:
        tts = PiperTTSService(
            voice_id="en_US-amy-medium",
        )
        print("[✓] Piper TTS initialized")
    except Exception as e:
        print(f"[✗] Piper TTS failed: {e}")
        return

    # ── Context ───────────────────────────────────────────────────
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        print("[✓] LLM Context aggregator initialized")
    except Exception as e:
        print(f"[✗] Context aggregator failed: {e}")
        return

    # ── Pipeline ──────────────────────────────────────────────────
    try:
        sentence_aggregator = SentenceAggregator()
        
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            sentence_aggregator,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        
        print("[✓] Pipeline constructed successfully!")
        print()
        print("[*] Creating pipeline task...")
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )
        
        print("[✓] Task created!")
        print()
        print("=" * 60)
        print("[READY] Voice pipeline is ready!")
        print("=" * 60)
        print()
        print("[*] Speak into your microphone...")
        print("[*] Press Ctrl+C to stop")
        print()
        
        runner = PipelineRunner()
        await runner.run(task)
        
    except KeyboardInterrupt:
        print("\n[*] Pipeline stopped by user")
    except Exception as e:
        print(f"[✗] Pipeline error: {e}")
        import traceback
        traceback.print_exc()

# ── Run ───────────────────────────────────────────────────────────
try:
    asyncio.run(test_pipeline())
except KeyboardInterrupt:
    print("\n[*] Test interrupted")
except Exception as e:
    print(f"\n[✗] Fatal error: {e}")
    import traceback
    traceback.print_exc()
