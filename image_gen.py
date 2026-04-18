"""
Image Generation — Emily Agent
SD WebUI API integration with auto-start
"""

import subprocess
import base64
import time
import os
import io
import logging

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


def _image_deps_available() -> tuple[bool, str]:
    """Check optional dependencies required for image generation."""
    missing = []
    if requests is None:
        missing.append("requests")
    if Image is None:
        missing.append("Pillow")
    if missing:
        return False, ", ".join(missing)
    return True, ""


# ── Config ────────────────────────────────────────────────────────────
SD_WEBUI_PATH = r"E:\Data\Packages\Stable Diffusion WebUI"
IMAGE_OUT_DIR = r"E:\Local_Voice\generated_images"
SD_PORTS      = [7860, 7861, 7862, 7863, 7864]
SD_API_URL    = "http://127.0.0.1:7860"   # default, auto-update hoga


def find_sd_url() -> str | None:
    """Saare ports check karo — jo kaam kare woh return karo."""
    ok, missing = _image_deps_available()
    if not ok:
        logger.warning("Image dependencies missing: %s", missing)
        return None
    for port in SD_PORTS:
        try:
            url  = f"http://127.0.0.1:{port}"
            resp = requests.get(f"{url}/internal/ping", timeout=3)
            if resp.status_code == 200:
                print(f"✅  SD WebUI found at port {port}")
                logger.info("SD WebUI found at: %s", url)
                return url
        except Exception:
            continue
    return None


def is_sd_running() -> bool:
    """Check karo — port auto-detect karo."""
    global SD_API_URL
    url = find_sd_url()
    if url:
        SD_API_URL = url
        return True
    return False


def start_sd_webui() -> bool:
    """SD WebUI automatically start karo."""
    if is_sd_running():
        print("✅  SD WebUI already running!")
        logger.info("SD WebUI already running")
        return True

    print("🚀  Starting SD WebUI... (~30-60 seconds)")
    logger.info("Auto-starting SD WebUI")

    bat_path = os.path.join(SD_WEBUI_PATH, "webui-user.bat")

    if not os.path.exists(bat_path):
        print(f"❌  webui-user.bat not found at: {bat_path}")
        logger.error("webui-user.bat not found: %s", bat_path)
        return False

    env = os.environ.copy()
    env["COMMANDLINE_ARGS"] = "--api --xformers --no-half-vae"

    subprocess.Popen(
        bat_path,
        cwd=SD_WEBUI_PATH,
        env=env,
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    print("⏳  Waiting", end="", flush=True)
    for _ in range(90):
        time.sleep(2)
        print(".", end="", flush=True)
        if is_sd_running():
            print("\n✅  SD WebUI ready!")
            logger.info("SD WebUI started successfully")
            return True

    print("\n❌  SD WebUI failed to start.")
    logger.error("SD WebUI failed to start in 180 seconds")
    return False


# ══════════════════════════════════════════════════════════════════════
# Image Generation
# ══════════════════════════════════════════════════════════════════════

def generate_image(prompt: str, llm_client, llm_model: str) -> str:
    """Emily → LLM enhanced prompt → SD WebUI API → Image"""
    ok, missing = _image_deps_available()
    if not ok:
        return (
            "Image feature is not installed, boss. "
            f"Missing: {missing}. "
            "Install with: pip install -r image_gen_requirements.txt"
        )

    # ── Step 0: SD WebUI check ────────────────────────────────────
    if not is_sd_running():
        print("⚠️  SD WebUI not running — auto starting...")
        success = start_sd_webui()
        if not success:
            return (
                "Sorry boss, SD WebUI could not start automatically. "
                "Please open Stability Matrix and click Launch manually."
            )

    # ── Step 1: LLM se prompt enhance karo ───────────────────────
    print("🎨  Enhancing prompt with LLM...")
    try:
        response = llm_client.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Stable Diffusion prompt engineer. "
                        "Convert the user's simple request into a detailed, vivid SD prompt. "
                        "Include: subject details, art style, lighting, mood, and quality tags "
                        "like (masterpiece, best quality, highly detailed, 8k). "
                        "Keep it under 100 words. "
                        "Return ONLY the prompt — no explanation, no extra text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Create SD prompt for: {prompt}"
                }
            ],
            think=False,
        )
        enhanced_prompt = response.message.content.strip()
        print(f"📝  Enhanced prompt: {enhanced_prompt}")
        logger.info("Enhanced prompt: %s", enhanced_prompt)
    except Exception as e:
        enhanced_prompt = prompt
        logger.warning("LLM enhance failed, using original: %s", e)
        print(f"⚠️  Using original prompt: {e}")

    # ── Step 2: SD WebUI API call ─────────────────────────────────
    print("🖼️   Generating image... (~15-30 seconds)")

    payload = {
        "prompt": enhanced_prompt,
        "negative_prompt": (
            "ugly, blurry, low quality, watermark, text, "
            "signature, bad anatomy, deformed, extra limbs, "
            "poorly drawn face, mutation, disfigured"
        ),
        "steps"       : 15,
        "cfg_scale"   : 6,
        "width"       : 512,
        "height"      : 512,
        "sampler_name": "DPM++ 2M",
        "batch_size"  : 1,
        "seed"        : -1,
    }

    try:
        resp = requests.post(
            f"{SD_API_URL}/sdapi/v1/txt2img",
            json=payload,
            timeout=600
        )

        if resp.status_code == 200:
            data     = resp.json()
            img_data = base64.b64decode(data["images"][0])
            img      = Image.open(io.BytesIO(img_data))

            os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
            filename    = f"emily_{int(time.time())}.png"
            output_path = os.path.join(IMAGE_OUT_DIR, filename)
            img.save(output_path)

            print(f"✅  Image saved: {output_path}")
            logger.info("Image saved: %s", output_path)

            os.startfile(output_path)
            return f"Image generated and opened boss! Saved as {filename}"

        else:
            logger.error("SD WebUI API error: %s", resp.status_code)
            return f"SD WebUI error: {resp.status_code} — {resp.text[:100]}"

    except requests.Timeout:
        return "Image generation timed out boss. Try a simpler prompt."
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return f"Image generation failed: {e}"
