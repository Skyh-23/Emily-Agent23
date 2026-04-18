"""
Intent Engine — Fast Hybrid Intent Classifier
===============================================
Classifies user input into one of 8 intents using keyword + pattern matching.
Works 100% offline, <50ms detection. No LLM required.

Intents:
  OPEN_APP       - open/launch/start an application
  CLOSE_APP      - close/quit/exit an application
  SEARCH_WEB     - search the web for something
  TYPE_TEXT      - type text into active window
  PLAY_MUSIC     - play music/songs
  CONTROL_SYSTEM - volume, brightness, screenshot, system controls
  GENERAL_CHAT   - normal conversation → goes to LLM
  UNKNOWN        - couldn't classify

Output:
  {"type": "OPEN_APP", "target": "chrome", "confidence": 0.95, "raw": "open chrome"}
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Intent type constants ────────────────────────────────────────────
OPEN_APP       = "OPEN_APP"
CLOSE_APP      = "CLOSE_APP"
SEARCH_WEB     = "SEARCH_WEB"
TYPE_TEXT      = "TYPE_TEXT"
PLAY_MUSIC     = "PLAY_MUSIC"
CONTROL_SYSTEM = "CONTROL_SYSTEM"
GENERAL_CHAT   = "GENERAL_CHAT"
UNKNOWN        = "UNKNOWN"

# ── Alias normalization ──────────────────────────────────────────────
# Maps natural language variations to canonical app names
APP_ALIASES = {
    "browser":        "chrome",
    "google chrome":  "chrome",
    "internet":       "chrome",
    "web browser":    "chrome",
    "edge browser":   "edge",
    "microsoft edge": "edge",
    "word":           "microsoft word",
    "ms word":        "microsoft word",
    "excel":          "microsoft excel",
    "ms excel":       "microsoft excel",
    "powerpoint":     "microsoft powerpoint",
    "ppt":            "microsoft powerpoint",
    "notepad++":      "notepad++",
    "vscode":         "visual studio code",
    "vs code":        "visual studio code",
    "code editor":    "visual studio code",
    "terminal":       "cmd",
    "command prompt": "cmd",
    "task manager":   "taskmgr",
    "file explorer":  "explorer",
    "files":          "explorer",
    "calculator":     "calc",
    "calc":           "calc",
    "paint":          "mspaint",
    "discord app":    "discord",
    "whatsapp app":   "whatsapp",
    "spotify app":    "spotify",
}

# ── Intent keyword rules ─────────────────────────────────────────────
# Order matters — checked top-to-bottom, first match wins
# Each rule: (intent, trigger_keywords, strip_prefixes)

_RULES = [

    # ── CLOSE_APP (check BEFORE OPEN — "close" is more specific) ─────
    (CLOSE_APP, [
        "close ", "shut down ", "exit ", "quit ", "kill ",
        "band karo ", "band kar ", "bund karo ", "bund kar ",
        "hatao ", "shut ",
        "close the ", "band kardo ",
    ], [
        "close ", "close the ", "shut down ", "shut ", "exit ", "quit ",
        "kill ", "band karo ", "band kar ", "bund karo ", "bund kar ",
        "hatao ", "band kardo ",
    ]),

    # ── OPEN_APP ──────────────────────────────────────────────────────
    (OPEN_APP, [
        "open ", "launch ", "start ", "run ",
        "kholo ", "khol do ", "khol ",
        "chalu karo ", "chalu kar ",
        "open the ", "start the ", "launch the ",
    ], [
        "open ", "open the ", "launch ", "launch the ",
        "start ", "start the ", "run ",
        "kholo ", "khol do ", "khol ",
        "chalu karo ", "chalu kar ",
    ]),

    # ── SEARCH_WEB ───────────────────────────────────────────────────
    (SEARCH_WEB, [
        "search ", "search for ", "google ", "google karo ",
        "look up ", "find online ", "dhundo ", "dhundho ",
        "google pe ", "google search ", "search karo ",
        "web search ", "bing ", "youtube search ",
    ], [
        "search for ", "search ", "google karo ", "google pe ",
        "google search ", "google ", "look up ", "find online ",
        "dhundo ", "dhundho ", "search karo ", "web search ",
        "bing search ", "bing ", "youtube search for ", "youtube search ",
    ]),

    # ── TYPE_TEXT ─────────────────────────────────────────────────────
    (TYPE_TEXT, [
        "type ", "write ", "likho ", "likh do ",
        "type this ", "type the text ", "enter text ",
        "type in ", "type out ",
    ], [
        "type this ", "type the text ", "type in ", "type out ",
        "type ", "write ", "likho ", "likh do ",
        "enter text ",
    ]),

    # ── PLAY_MUSIC ───────────────────────────────────────────────────
    (PLAY_MUSIC, [
        "play ", "play music", "play song", "gaana chalao",
        "music chalao", "chalao ", "music play", "song play",
        "next song", "previous song", "pause music", "resume music",
        "stop music",
    ], [
        "play music ", "play song ", "play the song ", "play the ",
        "play ", "gaana chalao ", "music chalao ", "chalao ",
    ]),

    # ── CONTROL_SYSTEM ───────────────────────────────────────────────
    (CONTROL_SYSTEM, [
        "volume ", "brightness ", "screenshot", "screen shot",
        "mute", "unmute", "increase volume", "decrease volume",
        "volume up", "volume down", "increase brightness",
        "decrease brightness", "take screenshot", "capture screen",
        "restart", "shutdown", "sleep mode", "lock screen",
        "lock ", "screen lock",
    ], []),   # no stripping needed — target IS the system action
]

# Words that signal the user wants to LEARN/DEFINE a command
_LEARN_PATTERNS = [
    r"when (?:i|we) say (.+?),?\s+(.+)",
    r"if (?:i|we) say (.+?),?\s+(?:then\s+)?(.+)",
    r"save command[:\s]+(.+?)\s*[=\-\u2192]+\s*(.+)",
    r"create shortcut[:\s]+(.+?)\s*[=\-\u2192]+\s*(.+)",
    r"jab main (.+?) (?:bolun|bolu|kahunga)[,\s]+(.+?)(?:\s+karo)?$",
    r"remember[:\s]+when i say (.+?),?\s+(.+)",
    r"teach[:\s]+(.+?)\s*[=\-\u2192]+\s*(.+)",
]

# ── Compound splitters ────────────────────────────────────────────────
_COMPOUND_SEPS = [
    " and then ", " then ", " aur phir ", " aur ", " and ",
    " phir ", ", then ", ", and ", " uske baad ",
]


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════

def detect_intent(text: str) -> dict:
    """
    Classify user input into an intent.

    Args:
        text: Raw user input

    Returns:
        {
          "type":       "OPEN_APP",
          "target":     "chrome",
          "confidence": 0.95,
          "raw":        "open chrome please"
        }
    """
    if not text or not text.strip():
        return _make(UNKNOWN, "", 0.0, text)

    normalized = _normalize(text)

    # ── Check learn/teach pattern first ────────────────────────────────
    learn = detect_learn_command(text)
    if learn:
        return _make("LEARN_COMMAND", "", 0.99, text, extra=learn)

    # ── Rule-based matching ─────────────────────────────────────────────
    for intent, triggers, strip_list in _RULES:
        for trigger in triggers:
            if normalized.startswith(trigger) or f" {trigger}" in f" {normalized}":
                target = _extract_target(normalized, strip_list, trigger)
                target = _resolve_alias(target)
                confidence = 0.95 if normalized.startswith(trigger) else 0.80
                logger.debug("Intent: %s → target='%s' (trigger='%s')", intent, target, trigger)
                return _make(intent, target, confidence, text)

    # ── Special: PLAY_MUSIC fallback ────────────────────────────────────
    if any(w in normalized for w in ["song", "music", "gaana", "gana", "playlist"]):
        return _make(PLAY_MUSIC, normalized, 0.70, text)

    # ── No match → GENERAL_CHAT ─────────────────────────────────────────
    return _make(GENERAL_CHAT, "", 0.60, text)


def detect_learn_command(text: str) -> Optional[dict]:
    """
    Detect if the user is trying to define a new custom command.

    Returns:
        {"trigger": "gaming mode", "actions": ["open steam", "open discord"]}
        or None if not a learn command
    """
    lower = text.lower().strip()
    for pattern in _LEARN_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            trigger = m.group(1).strip().strip("'\"")
            action_text = m.group(2).strip().strip("'\"")
            actions = _split_actions(action_text)
            if trigger and actions:
                return {"trigger": trigger, "actions": actions}
    return None


def split_compound(text: str) -> list[str]:
    """
    Split compound commands: "open steam and discord" → ["open steam", "open discord"]
    """
    lower = text.lower()
    for sep in sorted(_COMPOUND_SEPS, key=len, reverse=True):
        if sep in lower:
            parts = lower.split(sep)
            return [p.strip() for p in parts if p.strip()]
    return [text.strip()]


def is_system_command(text: str) -> bool:
    """Quick check: is this definitely NOT a general chat message?"""
    intent = detect_intent(text)
    return intent["type"] not in (GENERAL_CHAT, UNKNOWN)


# ── Private helpers ───────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, remove trailing punctuation."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.!?]+$", "", t)
    return t


def _extract_target(normalized: str, strip_list: list, matched_trigger: str) -> str:
    """Remove the command verb to get the target/payload."""
    result = normalized
    # Sort by length descending so we strip longest match first
    for prefix in sorted(strip_list, key=len, reverse=True):
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
            break
    # Remove courtesy words
    for word in ["please", "boss", "yaar", "bhai", "ji", "na", "for me"]:
        result = re.sub(rf"\b{re.escape(word)}\b", "", result).strip()
    return result.strip()


def _resolve_alias(target: str) -> str:
    """Map common aliases to canonical app names."""
    lower = target.lower().strip()
    if lower in APP_ALIASES:
        return APP_ALIASES[lower]
    # Partial match
    for alias, canonical in APP_ALIASES.items():
        if alias in lower:
            return canonical
    return target


def _split_actions(action_text: str) -> list[str]:
    """Split action text into individual actions."""
    actions = []
    for sep in sorted(_COMPOUND_SEPS, key=len, reverse=True):
        if sep in action_text:
            parts = action_text.split(sep)
            for p in parts:
                p = p.strip()
                if p:
                    actions.append(p)
            return actions
    return [action_text.strip()] if action_text.strip() else []


def _make(intent_type: str, target: str, confidence: float,
          raw: str, extra: dict = None) -> dict:
    result = {
        "type":       intent_type,
        "target":     target,
        "confidence": confidence,
        "raw":        raw,
    }
    if extra:
        result.update(extra)
    return result


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # Open app
        ("open chrome",               OPEN_APP,       "chrome"),
        ("launch visual studio code", OPEN_APP,       "visual studio code"),
        ("start notepad",             OPEN_APP,       "notepad"),
        ("kholo discord",             OPEN_APP,       "discord"),
        ("open the browser",          OPEN_APP,       "chrome"),       # alias
        # Close app
        ("close chrome",              CLOSE_APP,      "chrome"),
        ("band karo spotify",         CLOSE_APP,      "spotify"),
        ("exit notepad please",       CLOSE_APP,      "notepad"),
        # Search
        ("search for python tutorials", SEARCH_WEB,   "python tutorials"),
        ("google karo best phones",     SEARCH_WEB,   "best phones"),
        ("dhundo recipe for biryani",   SEARCH_WEB,   "recipe for biryani"),
        # Type text
        ("type hello world",          TYPE_TEXT,      "hello world"),
        ("likho mera naam Hiren hai", TYPE_TEXT,      "mera naam hiren hai"),
        # Music
        ("play Shape of You",         PLAY_MUSIC,     "shape of you"),
        ("gaana chalao",              PLAY_MUSIC,     ""),
        # System
        ("volume up",                 CONTROL_SYSTEM, ""),
        ("take screenshot",           CONTROL_SYSTEM, ""),
        # Learn
        ("when I say gaming mode, open steam and discord", "LEARN_COMMAND", ""),
        ("if I say work mode, then open vscode and chrome","LEARN_COMMAND", ""),
        # General
        ("what is the capital of India", GENERAL_CHAT, ""),
        ("tell me a joke",               GENERAL_CHAT, ""),
    ]

    passed = failed = 0
    print("\n" + "=" * 60)
    print("  Intent Engine — Self Test")
    print("=" * 60)
    for text, expected_type, expected_target in tests:
        result = detect_intent(text)
        ok = result["type"] == expected_type
        if expected_target and ok:
            ok = ok and (expected_target.lower() in result["target"].lower()
                         or result["target"].lower() in expected_target.lower())
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {text!r}")
        if not ok:
            print(f"         Expected: {expected_type!r} target={expected_target!r}")
            print(f"         Got:      {result['type']!r} target={result['target']!r}")
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)
