"""Quick mic test — tests all input devices to find the loudest one."""
import sounddevice as sd
import numpy as np

print("=" * 50)
print("MIC TEST — speak into your mic during recording!")
print("=" * 50)

for dev_id in [1, 2, 14, 15]:
    try:
        info = sd.query_devices(dev_id)
        if info["max_input_channels"] > 0:
            name = info["name"][:45]
            print(f"\nDevice {dev_id}: {name}")
            print("  Recording 3 seconds... SPEAK NOW!")
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32", device=dev_id)
            sd.wait()
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.abs(audio).max())
            print(f"  RMS={rms:.6f}  Peak={peak:.6f}")
            if peak > 0.01:
                print(f"  ✅ GOOD — this mic picks up your voice!")
            else:
                print(f"  ❌ LOW — barely picking up anything")
    except Exception as e:
        print(f"\nDevice {dev_id}: Error — {e}")

print("\n" + "=" * 50)
print("Pick the device with highest RMS/Peak for best results")
