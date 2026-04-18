"""
Memory Search Module — Semantic + Hybrid Ranking
==================================================
2-Stage retrieval:
  Stage 1: LanceDB vector search (22 candidates / 10 for short queries)
  Stage 2: Hybrid re-ranking: (semantic*0.6) + (recency*0.2) + (importance*0.2)
  Output:  Top 12 memories (top 5 for short queries)

Startup:   format_memories_for_context(limit=8) → 8 recent into system prompt
Per-query: smart_search_for_context() → up to 12 dynamically injected
"""

import logging
from datetime import datetime
from typing import Optional
from memory_store import (
    load_memories, get_memory, get_all_memories,
    semantic_search, count_memories as _count,
)

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
try:
    import config
    SEARCH_LIMIT         = getattr(config, "MEMORY_SEARCH_LIMIT", 12)
    SIMILARITY_THRESHOLD = getattr(config, "MEMORY_SIMILARITY_THRESHOLD", 0.3)
    RECENCY_WEIGHT       = getattr(config, "MEMORY_RECENCY_WEIGHT", 0.2)
    SEMANTIC_WEIGHT      = getattr(config, "MEMORY_SEMANTIC_WEIGHT", 0.6)
    IMPORTANCE_WEIGHT    = getattr(config, "MEMORY_IMPORTANCE_WEIGHT", 0.2)
except ImportError:
    SEARCH_LIMIT         = 12
    SIMILARITY_THRESHOLD = 0.3
    RECENCY_WEIGHT       = 0.2
    SEMANTIC_WEIGHT      = 0.6
    IMPORTANCE_WEIGHT    = 0.2


# ── Basic search helpers ─────────────────────────────────────────────

def search_by_id(memory_id: str) -> Optional[dict]:
    return get_memory(memory_id)


def search_by_content(query: str, case_sensitive: bool = False) -> list:
    memories = get_all_memories()
    q = query if case_sensitive else query.lower()
    return [m for m in memories
            if q in (m["content"] if case_sensitive else m["content"].lower())]


def search_by_tag(tag: str) -> list:
    memories = get_all_memories()
    return [m for m in memories if tag.lower() in [t.lower() for t in m.get("tags", [])]]


def search_recent(limit: int = 10, show_ui: bool = False) -> list:
    """Get most recently created memories (newest first)."""
    memories = get_all_memories()
    sorted_mem = sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True)
    results = sorted_mem[:limit]

    if show_ui and results:
        print(f"\n📚 Recent {len(results)} memories:")
        print("─" * 50)
        for i, mem in enumerate(results, 1):
            content = mem.get("content", "").strip()
            if "\nEmily: " in content and content.startswith("User: "):
                parts   = content.split("\nEmily: ", 1)
                preview = parts[0].replace("User: ", "", 1).strip()[:40]
            else:
                preview = content[:50]
            created = mem.get("created_at", "")[:10]
            print(f"  {i}. {preview}...")
            print(f"     📅 {created} | ID: {mem.get('id','?')[:8]}...")
        print("─" * 50 + "\n")

    return results


def count_memories() -> int:
    return _count()


def get_memory_summary() -> dict:
    try:
        return {"status": "connected", "total_memories": count_memories(), "engine": "LanceDB"}
    except Exception as e:
        return {"status": "error", "total_memories": 0, "error": str(e)}


# ── Hybrid Ranking helpers ───────────────────────────────────────────

def _recency_score(created_at: str, decay_days: int = 30) -> float:
    """1.0 = today, decays to 0.0 over decay_days."""
    try:
        created  = datetime.fromisoformat(created_at)
        age_days = (datetime.now() - created).total_seconds() / 86400.0
        return max(0.0, 1.0 - (age_days / decay_days))
    except Exception:
        return 0.5


def _hybrid_score(similarity: float, recency: float, importance: float) -> float:
    """(semantic*0.6) + (recency*0.2) + (importance_norm*0.2)"""
    imp_norm = min(importance / 2.0, 1.0)
    return (SEMANTIC_WEIGHT * similarity
            + RECENCY_WEIGHT * recency
            + IMPORTANCE_WEIGHT * imp_norm)


def _is_short_query(query: str) -> bool:
    words = query.strip().split()
    return len(query.strip()) < 20 or len(words) < 3


# ── Smart Search — 2-Stage Re-Ranking ───────────────────────────────

def smart_search_for_context(user_message: str, limit: int = None,
                              exclude_ids: set = None, show_ui: bool = True) -> str:
    """
    2-Stage memory retrieval with hybrid re-ranking.

    Stage 1: LanceDB vector search — 22 candidates (10 for short queries)
    Stage 2: Re-rank: final = (semantic*0.6) + (recency*0.2) + (importance*0.2)
    Output:  Top 12 memories (top 5 for short queries)

    Returns formatted string to inject into LLM context, or "" if nothing found.
    """
    if not user_message or not user_message.strip():
        return ""

    exclude_ids = exclude_ids or set()
    short_query = _is_short_query(user_message)
    fetch_limit = 10 if short_query else 22
    final_limit = limit or (5 if short_query else SEARCH_LIMIT)

    try:
        # Stage 1: vector search
        raw_results = semantic_search(
            query=user_message,
            limit=fetch_limit,
            threshold=SIMILARITY_THRESHOLD,
        )

        if not raw_results:
            logger.debug("No memory candidates for: %s", user_message[:40])
            return ""

        # Stage 2: hybrid re-ranking
        scored = []
        for mem in raw_results:
            if mem["id"] in exclude_ids:
                continue
            similarity = mem.get("_similarity", 0.5)
            recency    = _recency_score(mem.get("created_at", ""))
            importance = mem.get("importance", 1.0)
            hybrid     = _hybrid_score(similarity, recency, importance)
            scored.append((hybrid, similarity, mem))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top_memories = scored[:final_limit]

        # Terminal feedback
        if show_ui and top_memories:
            tag = " [short→5]" if short_query else ""
            print(f"🔍 Memory search{tag}: {len(raw_results)} candidates → {len(top_memories)} selected")
            for i, (h, sim, mem) in enumerate(top_memories[:3], 1):
                content = mem.get("content", "").strip()
                if "\nEmily: " in content and content.startswith("User: "):
                    preview = content.split("\nEmily: ")[0].replace("User: ", "", 1)[:40]
                else:
                    preview = content[:40]
                print(f"  {i}. [{int(sim*100)}% sim | {h:.2f}] {preview}...")
            if len(top_memories) > 3:
                print(f"  ... +{len(top_memories) - 3} more")

        # Format for LLM injection — clean, no meta tags
        lines = []
        for _, _, mem in top_memories:
            content = mem.get("content", "").strip()
            if not content:
                continue
            if "\nEmily: " in content and content.startswith("User: "):
                parts      = content.split("\nEmily: ", 1)
                user_part  = parts[0].replace("User: ", "", 1).strip()
                emily_part = parts[1].strip()
                truncated  = emily_part[:100] + ("..." if len(emily_part) > 100 else "")
                lines.append(f'- User said: "{user_part}" → You replied: "{truncated}"')
            else:
                lines.append(f"- {content[:150]}")

        if not lines:
            return ""

        logger.info("Injecting %d memories for: '%s'", len(lines), user_message[:40])

        return (
            "RELEVANT MEMORY — You MUST use this information to answer. "
            "These are real past conversations you had with this user. "
            "NEVER say 'I don't know' if the answer is here:\n"
            + "\n".join(lines)
        )

    except Exception as e:
        logger.error("Smart search failed: %s", e)
        return ""


# ── Startup Context (8 Recent Memories) ─────────────────────────────

def format_memories_for_context(limit: int = 8) -> str:
    """
    Format N most recent memories for system prompt at startup.
    Per-query retrieval is handled by smart_search_for_context().
    """
    recent = search_recent(limit=limit)
    if not recent:
        return ""

    lines = []
    for mem in reversed(recent):   # oldest first = natural order
        content = mem.get("content", "").strip()
        if not content:
            continue
        if "\nEmily: " in content and content.startswith("User: "):
            parts      = content.split("\nEmily: ", 1)
            user_part  = parts[0].replace("User: ", "", 1).strip()
            emily_part = parts[1].strip()
            truncated  = emily_part[:100] + ("..." if len(emily_part) > 100 else "")
            lines.append(f'- User said: "{user_part}" → You replied: "{truncated}"')
        else:
            lines.append(f"- {content[:150]}")

    if not lines:
        return ""

    return (
        "YOUR MEMORY — These are REAL past conversations you had with this user. "
        "You MUST remember and reference this information when the user asks about "
        "things you discussed before. NEVER say 'I don't know' or give generic answers "
        "if the answer is in your memory:\n"
        + "\n".join(lines)
    )


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = get_memory_summary()
    print(f"Status : {summary['status'].upper()} | Engine: {summary.get('engine')} | Total: {summary['total_memories']}")
    print("\n--- Startup context (8 recent) ---")
    ctx = format_memories_for_context(limit=8)
    print(ctx if ctx else "(empty)")
    print("\n--- Search: 'favorite color' ---")
    smart = smart_search_for_context("what is my favorite color?", show_ui=True)
    print(smart if smart else "(no matches)")
    print("\n--- Short query: 'hi' ---")
    s2 = smart_search_for_context("hi", show_ui=True)
    print(s2 if s2 else "(short query - minimal)")
