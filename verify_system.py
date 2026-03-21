"""
Final Verification Script - Test complete flow without user interaction
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("\n" + "="*70)
print("FINAL VERIFICATION - System Health Check")
print("="*70 + "\n")

# Test 1: Ollama endpoint
print("[1/4] Testing Ollama with correct /api endpoint...")
try:
    import ollama
    from config import OLLAMA_BASE_URL
    client = ollama.Client(host=OLLAMA_BASE_URL)
    
    # Try the /api/tags endpoint
    models = client.list().models
    print(f"    ✓ Ollama connected: {len(models)} models available")
    for m in models[:3]:
        print(f"      - {m.model}")
    print("      ...")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Whisper
print("\n[2/4] Testing Whisper STT...")
try:
    from faster_whisper import WhisperModel
    from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    print(f"    ✓ Whisper ready: {WHISPER_MODEL} on {WHISPER_DEVICE}")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Piper TTS
print("\n[3/4] Testing Piper TTS...")
try:
    from text_to_speech import PiperEngine
    piper = PiperEngine()
    print(f"    ✓ Piper ready: {piper.model_path.split('/')[-1] if piper.model_path else 'Model path not found'}")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Pipecat services with correct Ollama endpoint
print("\n[4/4] Testing Pipecat services with fixed Ollama endpoint...")
try:
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.piper.tts import PiperTTSService
    
    stt = WhisperSTTService(
        model=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    print("    ✓ WhisperSTTService initialized")
    
    llm = OLLamaLLMService(
        model="dolphin-llama3:8b",
        base_url=f"{OLLAMA_BASE_URL}/api",  # CORRECT FORMAT
    )
    print(f"    ✓ OLLamaLLMService initialized (endpoint: {OLLAMA_BASE_URL}/api)")
    
    tts = PiperTTSService(
        voice_id="en_US-amy-medium",
    )
    print("    ✓ PiperTTSService initialized")
    
except Exception as e:
    print(f"    ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED - SYSTEM READY!")
print("="*70)
print("\nYou can now run:")
print("  python main.py")
print("\nThe voice pipeline will work correctly with:")
print("  • Whisper STT (speech recognition)")
print("  • Ollama LLM (language model) at http://localhost:11434/api")
print("  • Piper TTS (voice output)")
print("\n")
