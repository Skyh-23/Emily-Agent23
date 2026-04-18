"""
Custom Commands — Self-Learning Command System
===============================================
Allows users to define custom voice triggers dynamically.

Example:
  User: "When I say gaming mode, open steam and discord"
  Emily saves trigger='gaming mode' → actions=['open steam', 'open discord']
  Next time: "gaming mode" → executes instantly, skips LLM

Storage:
  - Primary: Conversation/custom_commands.json  (instant load)
  - Mirror:  LanceDB (tagged 'custom_command' for search)

Safety:
  - Protected triggers cannot be overwritten
  - Optional confirmation before saving
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
_DIR  = os.path.join(os.path.dirname(__file__), "Conversation")
_FILE = os.path.join(_DIR, "custom_commands.json")

# ── Protected triggers — these cannot be overwritten ─────────────────
PROTECTED_TRIGGERS = {
    "open", "close", "type", "search", "play", "stop",
    "launch", "exit", "quit", "run", "start",
}

# Fuzzy match threshold (0-100): how similar trigger must be to user input
_MATCH_THRESHOLD = 82

# ── In-memory cache ───────────────────────────────────────────────────
_commands: dict[str, dict] = {}   # trigger_normalized → {trigger, actions, created_at}
_loaded = False


# ══════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════

def _ensure_dir():
    os.makedirs(_DIR, exist_ok=True)


def load_commands() -> dict:
    """Load all custom commands from JSON. Returns {trigger: {actions, ...}}"""
    global _commands, _loaded
    if _loaded:
        return _commands

    _ensure_dir()
    if os.path.exists(_FILE):
        try:
            with open(_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _commands = {_norm(k): v for k, v in data.items()}
            logger.info("Loaded %d custom commands from %s", len(_commands), _FILE)
            if _commands:
                print(f"🎯  Custom commands: {len(_commands)} loaded "
                      f"({', '.join(list(_commands.keys())[:5])}{'...' if len(_commands) > 5 else ''})")
        except Exception as e:
            logger.warning("Could not load custom commands: %s", e)
            _commands = {}
    else:
        _commands = {}

    _loaded = True
    return _commands


def _save_to_disk():
    """Persist commands dict to JSON."""
    _ensure_dir()
    try:
        with open(_FILE, "w", encoding="utf-8") as f:
            json.dump(_commands, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Could not save custom commands: %s", e)


def _mirror_to_lancedb(trigger: str, actions: list[str]):
    """Also store in LanceDB for completeness (best-effort)."""
    try:
        from memory_store import add_memory
        content = f"Custom command: '{trigger}' → {', '.join(actions)}"
        add_memory(content, tags=["custom_command"], metadata={"trigger": trigger})
    except Exception as e:
        logger.debug("LanceDB mirror failed (non-critical): %s", e)


# ══════════════════════════════════════════════════════════════════════
# Command Management
# ══════════════════════════════════════════════════════════════════════

def save_command(trigger: str, actions: list[str],
                 confirm_callback=None) -> tuple[bool, str]:
    """
    Save a custom command.

    Args:
        trigger:          The phrase that activates this command
        actions:          List of action strings e.g. ['open steam', 'open discord']
        confirm_callback: Optional callable(msg) → bool for confirmation UI

    Returns:
        (success: bool, message: str)
    """
    global _commands
    load_commands()

    trigger_norm = _norm(trigger)

    # Safety: protect critical triggers
    if trigger_norm in PROTECTED_TRIGGERS or any(
        trigger_norm == p for p in PROTECTED_TRIGGERS
    ):
        return False, f"Cannot save — '{trigger}' is a protected system command."

    if not actions:
        return False, "No actions provided for the command."

    # Check if overwriting existing
    if trigger_norm in _commands:
        existing = _commands[trigger_norm]["actions"]
        if existing == actions:
            return True, f"Command '{trigger}' already exists with the same actions."
        # Ask for confirmation if overwriting
        if confirm_callback:
            ok = confirm_callback(
                f"Command '{trigger}' already exists ({', '.join(existing)}). Overwrite?"
            )
            if not ok:
                return False, "Overwrite cancelled."

    # Save
    _commands[trigger_norm] = {
        "trigger":    trigger,
        "trigger_norm": trigger_norm,
        "actions":    actions,
        "created_at": datetime.now().isoformat(),
        "use_count":  0,
    }
    _save_to_disk()
    _mirror_to_lancedb(trigger, actions)

    action_str = " + ".join(actions)
    logger.info("Custom command saved: '%s' → %s", trigger, actions)
    return True, f"Saved! '{trigger}' will now → {action_str}"


def delete_command(trigger: str) -> tuple[bool, str]:
    """Delete a custom command by trigger."""
    global _commands
    load_commands()
    key = _norm(trigger)
    if key not in _commands:
        return False, f"No command found for '{trigger}'."
    del _commands[key]
    _save_to_disk()
    return True, f"Deleted command '{trigger}'."


def list_commands() -> list[dict]:
    """Return all saved custom commands as a list."""
    load_commands()
    return list(_commands.values())


# ══════════════════════════════════════════════════════════════════════
# Matching
# ══════════════════════════════════════════════════════════════════════

def match_command(text: str) -> Optional[dict]:
    """
    Check if user input matches any saved custom command.
    Priority: exact → whole-word substring → fuzzy (threshold 82)
    """
    load_commands()
    if not _commands:
        return None

    normalized = _norm(text)

    # 1. Exact match
    if normalized in _commands:
        cmd = _commands[normalized]
        _increment_use(normalized)
        return cmd

    # 2. Whole-trigger contained in input (e.g. "work mode please")
    #    Only if trigger is meaningful (>=4 chars) to avoid single-word false matches
    for key, cmd in _commands.items():
        if len(key) >= 4 and re.search(rf"\b{re.escape(key)}\b", normalized):
            _increment_use(key)
            return cmd

    # 3. Fuzzy match — only apply when input length is close to trigger length
    best_score = 0
    best_key   = None
    for key in _commands:
        # Skip if length difference is too large (avoids "open chrome" matching "gaming mode")
        if abs(len(normalized) - len(key)) > max(len(key) // 2, 3):
            continue
        score = _simple_ratio(normalized, key)
        if score > best_score:
            best_score = score
            best_key   = key

    if best_score >= _MATCH_THRESHOLD and best_key:
        _increment_use(best_key)
        return _commands[best_key]

    return None


def _increment_use(key: str):
    """Track usage count."""
    global _commands
    if key in _commands:
        _commands[key]["use_count"] = _commands[key].get("use_count", 0) + 1
        # Periodically save (every 5 uses)
        if _commands[key]["use_count"] % 5 == 0:
            _save_to_disk()


# ══════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════

def execute_command(cmd: dict, executor=None) -> str:
    """
    Execute all actions in a custom command.

    Args:
        cmd:      The command dict (from match_command)
        executor: Optional CommandExecutor instance

    Returns:
        Result string for TTS
    """
    actions = cmd.get("actions", [])
    trigger = cmd.get("trigger", "command")

    if not actions:
        return f"Command '{trigger}' has no actions defined."

    if executor is None:
        try:
            from commands import CommandExecutor
            executor = CommandExecutor()
        except Exception as e:
            return f"Could not execute command: {e}"

    results = []
    for action in actions:
        action = action.strip()
        result = _dispatch_action(action, executor)
        results.append(result)
        logger.info("Custom cmd '%s' → action '%s' → %s", trigger, action, result)

    return f"Done boss! Executed '{trigger}': " + " | ".join(results)


def _dispatch_action(action: str, executor) -> str:
    """Route a single action string to the right executor method."""
    from intent_engine import detect_intent, OPEN_APP, CLOSE_APP, SEARCH_WEB, TYPE_TEXT

    intent = detect_intent(action)
    itype  = intent["type"]
    target = intent["target"] or action

    try:
        if itype == OPEN_APP:
            return executor.open_application(target)
        elif itype == CLOSE_APP:
            return executor.close_application(target)
        elif itype == SEARCH_WEB:
            import webbrowser
            webbrowser.open(f"https://www.google.com/search?q={target.replace(' ', '+')}")
            return f"Searched for {target}"
        elif itype == TYPE_TEXT:
            return executor.type_text(target)
        else:
            # Fallback: try open_application with raw action
            return executor.open_application(action)
    except Exception as e:
        logger.warning("Action dispatch failed for '%s': %s", action, e)
        return f"Could not execute '{action}'"


# ══════════════════════════════════════════════════════════════════════
# NL Learning Parser
# ══════════════════════════════════════════════════════════════════════

def parse_learn_intent(text: str) -> Optional[dict]:
    """
    Parse 'When I say X, do Y and Z' into trigger + actions.
    Returns {"trigger": str, "actions": [str, ...]} or None.
    """
    from intent_engine import detect_learn_command
    return detect_learn_command(text)


# ── Private helpers ───────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Normalize for storage/matching."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.!?,]+$", "", t)
    return t


def _simple_ratio(a: str, b: str) -> int:
    """
    Word-level Jaccard similarity (0-100).
    "gaming mode" vs "open chrome" → 0 shared words → 0%
    "gaming mode" vs "gaming mode" → 2/2 shared words → 100%
    Much more accurate than character overlap for trigger matching.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0
    intersection = words_a & words_b
    union = words_a | words_b
    return int((len(intersection) / len(union)) * 100)



# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Custom Commands — Self Test")
    print("=" * 55)

    # Test saving
    ok, msg = save_command("gaming mode", ["open steam", "open discord"])
    print(f"  Save: {msg}")

    ok, msg = save_command("work mode", ["open visual studio code", "open chrome"])
    print(f"  Save: {msg}")

    ok, msg = save_command("study mode", ["open notion", "play lo-fi music"])
    print(f"  Save: {msg}")

    # Test matching
    print("\n  --- Match Tests ---")
    tests = [
        ("gaming mode",      True),
        ("gaming",           False),   # too short for fuzzy at default threshold
        ("work mode please", True),    # substring match
        ("study mode",       True),
        ("open chrome",      False),   # this is a system command, not custom
    ]
    for user_input, should_match in tests:
        result = match_command(user_input)
        matched = result is not None
        status  = "PASS" if matched == should_match else "FAIL"
        print(f"  [{status}] '{user_input}' → matched={matched} "
              f"(expected={should_match})"
              + (f" → {result['actions']}" if result else ""))

    # Test protected
    print("\n  --- Protection Tests ---")
    ok, msg = save_command("open", ["open chrome"])
    print(f"  Protected 'open': {'PASS' if not ok else 'FAIL'} — {msg}")

    # Test list
    cmds = list_commands()
    print(f"\n  Stored commands: {len(cmds)}")
    for c in cmds:
        print(f"    '{c['trigger']}' → {c['actions']}")

    print("=" * 55)
