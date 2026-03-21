"""
Kokoro Hindi TTS Test — Correct lang_code
Run: python test_kokoro.py
"""

import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os
from kokoro import KPipeline

# ── Output folder ─────────────────────────────────────────────────────
OUTPUT_DIR = r"E:\Local_Voice\Test1_01"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁  Output: {OUTPUT_DIR}\n")

# ── Load Models ───────────────────────────────────────────────────────
print("🔄  Loading Kokoro Hindi pipeline (lang_code=h)...")
t0 = time.time()
pipeline_hindi = KPipeline(lang_code='h')   # ← Hindi ke liye 'h'
print(f"✅  Hindi pipeline loaded in {time.time()-t0:.1f}s\n")

print("🔄  Loading Kokoro English pipeline (lang_code=a)...")
t0 = time.time()
pipeline_english = KPipeline(lang_code='a') # ← English ke liye 'a'
print(f"✅  English pipeline loaded in {time.time()-t0:.1f}s\n")


# ── Speak Function ────────────────────────────────────────────────────
def speak(text: str, voice: str, pipeline, filename: str, speed: float = 1.0):
    print(f"  🗣️  [{voice}] speed={speed}")
    print(f"  📝  {text[:70]}...")
    t0 = time.time()

    audio_parts = []
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is not None:
            audio_parts.append(result.audio.numpy())

    if not audio_parts:
        print("  ❌  No audio generated!\n")
        return

    full_audio = np.concatenate(audio_parts)
    elapsed = time.time() - t0
    duration = len(full_audio) / 24000

    wav_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(wav_path, full_audio, 24000)
    print(f"  ⚡  {elapsed:.2f}s generated  |  {duration:.1f}s audio")
    print(f"  💾  {filename}\n")

    sd.play(full_audio, 24000)
    sd.wait()


# ── Test Texts ────────────────────────────────────────────────────────
HINDI_TEXT = "हवेली का बंद दरवाजा। रात के 2 बज रहे थे। आंधी तूफान के कारण हवेली की खिड़कियां जोर जोर से खटखटा रही थीं। आर्यन अपनी टॉर्च लेकर हवेली के पुराने गलियारे से गुजर रहा था, तभी उसे लगा कि कोई उसके पीछे चल रहा है। वह रुका, पीछे मुड़ा, पर वहां कोई नहीं था।"

HINGLISH_TEXT = "रात के 3 बज रहे थे और Hostel की उस पुरानी building में lights अचानक चली गई, बस मेरे phone की screen जल रही थी, तभी मुझे लगा कोई मेरे पीछे खड़ा है।"

ENGLISH_TEXT = "Hello boss! It was two in the night. The wind was howling and the windows were rattling. Suddenly someone knocked on the door. I turned around but there was nobody there."


# ══════════════════════════════════════════════════════════════════════
# HINDI VOICES — lang_code='h'
# ══════════════════════════════════════════════════════════════════════

print("=" * 55)
print("TEST 1 — hf_alpha | Hindi Text")
print("=" * 55)
speak(HINDI_TEXT, "hf_alpha", pipeline_hindi, "alpha_Hindi.wav")

print("=" * 55)
print("TEST 2 — hf_alpha | Hinglish Text")
print("=" * 55)
speak(HINGLISH_TEXT, "hf_alpha", pipeline_hindi, "alpha_Hinglish.wav")

print("=" * 55)
print("TEST 3 — hf_alpha | English Text")
print("=" * 55)
speak(ENGLISH_TEXT, "hf_alpha", pipeline_hindi, "alpha_English.wav")

print("=" * 55)
print("TEST 4 — hf_beta | Hindi Text")
print("=" * 55)
speak(HINDI_TEXT, "hf_beta", pipeline_hindi, "beta_Hindi.wav")

print("=" * 55)
print("TEST 5 — hf_beta | Hinglish Text")
print("=" * 55)
speak(HINGLISH_TEXT, "hf_beta", pipeline_hindi, "beta_Hinglish.wav")

print("=" * 55)
print("TEST 6 — hf_beta | English Text")
print("=" * 55)
speak(ENGLISH_TEXT, "hf_beta", pipeline_hindi, "beta_English.wav")

print("=" * 55)
print("TEST 7 — hm_omega | Hindi Text")
print("=" * 55)
speak(HINDI_TEXT, "hm_omega", pipeline_hindi, "omega_Hindi.wav")

print("=" * 55)
print("TEST 8 — hm_omega | Hinglish Text")
print("=" * 55)
speak(HINGLISH_TEXT, "hm_omega", pipeline_hindi, "omega_Hinglish.wav")

print("=" * 55)
print("TEST 9 — hm_psi | Hindi Text")
print("=" * 55)
speak(HINDI_TEXT, "hm_psi", pipeline_hindi, "psi_Hindi.wav")

print("=" * 55)
print("TEST 10 — hm_psi | Hinglish Text")
print("=" * 55)
speak(HINGLISH_TEXT, "hm_psi", pipeline_hindi, "psi_Hinglish.wav")

# ══════════════════════════════════════════════════════════════════════
# BONUS — English pipeline se af_bella Hinglish compare
# ══════════════════════════════════════════════════════════════════════

print("=" * 55)
print("BONUS — af_bella | Hinglish (English pipeline)")
print("=" * 55)
speak(HINGLISH_TEXT, "af_bella", pipeline_english, "bella_Hinglish_compare.wav")

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 55)
print("✅  All done!")
print("=" * 55)
print("""
Files generated:
  📄  alpha_Hindi.wav        📄  alpha_Hinglish.wav    📄  alpha_English.wav
  📄  beta_Hindi.wav         📄  beta_Hinglish.wav     📄  beta_English.wav
  📄  omega_Hindi.wav        📄  omega_Hinglish.wav
  📄  psi_Hindi.wav          📄  psi_Hinglish.wav
  📄  bella_Hinglish_compare.wav
""")