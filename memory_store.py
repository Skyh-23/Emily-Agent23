"""
Memory Store Module — LanceDB + Parquet
========================================
Primary storage: LanceDB (vector-indexed, semantic search ready)
Backup: Parquet file (flat-file export)
Both text mode and voice mode use the SAME database.
"""

import json
import random
import os
import logging
import pyarrow as pa
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
try:
    import config
    MEMORY_ENABLED = getattr(config, "MEMORY_ENABLED", True)
    MEMORY_DB_PATH = getattr(config, "MEMORY_DB_PATH", "Conversation/memories.lance")
    MEMORY_PARQUET_PATH = getattr(config, "MEMORY_PARQUET_PATH", "Conversation/memories_backup.parquet")
except ImportError:
    MEMORY_ENABLED = True
    MEMORY_DB_PATH = "Conversation/memories.lance"
    MEMORY_PARQUET_PATH = "Conversation/memories_backup.parquet"

# Resolve paths relative to this file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DB_PATH = os.path.join(_BASE_DIR, MEMORY_DB_PATH)
MEMORY_PARQUET_PATH = os.path.join(_BASE_DIR, MEMORY_PARQUET_PATH)
MEMORY_FOLDER = os.path.join(_BASE_DIR, "Conversation")
LEGACY_JSON = os.path.join(MEMORY_FOLDER, "memories_boss.json")

# ── LanceDB connection (lazy singleton) ─────────────────────────────
_lance_db = None
_lance_table = None
TABLE_NAME = "memories"


def is_memory_enabled() -> bool:
    return MEMORY_ENABLED


def _ensure_folder():
    os.makedirs(MEMORY_FOLDER, exist_ok=True)


def generate_id() -> str:
    """Generate unique ID: hs-XXXXXXXXXX"""
    digits = random.randint(10, 12)
    number = ''.join([str(random.randint(0, 9)) for _ in range(digits)])
    return f"hs-{number}"


def _get_db():
    global _lance_db
    if _lance_db is None:
        import lancedb
        _ensure_folder()
        _lance_db = lancedb.connect(MEMORY_DB_PATH)
        logger.info("LanceDB connected: %s", MEMORY_DB_PATH)
    return _lance_db


def _get_table():
    global _lance_table
    if _lance_table is not None:
        return _lance_table

    db = _get_db()
    try:
        _lance_table = db.open_table(TABLE_NAME)
        logger.debug("Opened existing LanceDB table '%s'", TABLE_NAME)
    except Exception:
        _lance_table = None
        logger.info("LanceDB table '%s' not found, will create on first write.", TABLE_NAME)

    return _lance_table


def _create_table_with_data(data: list[dict]):
    global _lance_table
    db = _get_db()
    _lance_table = db.create_table(TABLE_NAME, data=data, mode="overwrite")
    logger.info("Created LanceDB table '%s' with %d records", TABLE_NAME, len(data))
    return _lance_table


# ── CRUD Operations ─────────────────────────────────────────────────

def add_memory(content: str, tags: Optional[list] = None,
               metadata: Optional[dict] = None,
               importance: float = 1.0) -> str:
    """
    Add a new memory to LanceDB with embedding vector.

    Returns:
        The generated memory ID (empty string if disabled)
    """
    if not is_memory_enabled():
        return ""

    from embedding_model import embed_text

    memory_id = generate_id()
    now = datetime.now().isoformat()

    record = {
        "id": memory_id,
        "content": content,
        "tags": ",".join(tags) if tags else "",
        "metadata_str": json.dumps(metadata or {}),
        "vector": embed_text(content),
        "created_at": now,
        "importance": importance,
    }

    try:
        table = _get_table()
        if table is None:
            _create_table_with_data([record])
            print(f"🧠 Memory saved (new table): {memory_id}")
        else:
            table.add([record])
            print(f"🧠 Memory saved: {memory_id} | {content[:40]}...")

        logger.info("Memory saved [%s] %s", memory_id, content[:60])

        # Mirror to graph (background thread, fire-and-forget — never blocks)
        try:
            from graph_bridge import sync_memory_to_graph_async
            sync_memory_to_graph_async(memory_id, content, tags or [], metadata or {}, now, importance)
        except Exception:
            pass  # Graph is optional — silent if unavailable

        _export_parquet()

    except Exception as e:
        print(f"❌ Memory save FAILED: {e}")
        logger.error("Failed to save memory %s: %s", memory_id, e)

    return memory_id


def get_all_memories() -> list[dict]:
    """Get all memories as a list of dicts."""
    try:
        table = _get_table()
        if table is None:
            return []
        df = table.to_pandas()
        records = []
        for _, row in df.iterrows():
            records.append({
                "id": row["id"],
                "content": row["content"],
                "tags": [t for t in row.get("tags", "").split(",") if t],
                "metadata": json.loads(row.get("metadata_str", "{}")),
                "created_at": row.get("created_at", ""),
                "importance": row.get("importance", 1.0),
            })
        return records
    except Exception as e:
        logger.error("Failed to get all memories: %s", e)
        return []


def get_memory(memory_id: str) -> Optional[dict]:
    """Get a single memory by ID."""
    try:
        table = _get_table()
        if table is None:
            return None
        results = table.search().where(f"id = '{memory_id}'").limit(1).to_list()
        if results:
            row = results[0]
            return {
                "id": row["id"],
                "content": row["content"],
                "tags": [t for t in row.get("tags", "").split(",") if t],
                "metadata": json.loads(row.get("metadata_str", "{}")),
                "created_at": row.get("created_at", ""),
                "importance": row.get("importance", 1.0),
            }
        return None
    except Exception as e:
        logger.error("Failed to get memory %s: %s", memory_id, e)
        return None


def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID."""
    if not is_memory_enabled():
        return False
    try:
        table = _get_table()
        if table is None:
            return False
        table.delete(f"id = '{memory_id}'")
        logger.info("Deleted memory %s", memory_id)
        _export_parquet()
        return True
    except Exception as e:
        logger.error("Failed to delete memory %s: %s", memory_id, e)
        return False


def count_memories() -> int:
    """Get total count of memories."""
    try:
        table = _get_table()
        if table is None:
            return 0
        return table.count_rows()
    except Exception as e:
        logger.error("Failed to count memories: %s", e)
        return 0


def semantic_search(query: str, limit: int = 12,
                    threshold: float = 0.3) -> list[dict]:
    """
    Vector similarity search using LanceDB.

    Returns:
        List of memories with '_similarity' field (higher = more similar)
    """
    try:
        table = _get_table()
        if table is None:
            return []

        from embedding_model import embed_text
        query_vector = embed_text(query)

        results = (
            table.search(query_vector)
            .limit(limit)
            .to_list()
        )

        memories = []
        for row in results:
            distance = row.get("_distance", 999)
            similarity = 1.0 / (1.0 + distance)

            if similarity >= threshold:
                memories.append({
                    "id": row["id"],
                    "content": row["content"],
                    "tags": [t for t in row.get("tags", "").split(",") if t],
                    "metadata": json.loads(row.get("metadata_str", "{}")),
                    "created_at": row.get("created_at", ""),
                    "importance": row.get("importance", 1.0),
                    "_similarity": similarity,
                    "_distance": distance,
                })

        logger.debug("Semantic search for '%s': %d results", query[:30], len(memories))
        return memories

    except Exception as e:
        logger.error("Semantic search failed: %s", e)
        return []


# ── Parquet Export ───────────────────────────────────────────────────

def _export_parquet():
    """Export current memories to Parquet backup file."""
    try:
        table = _get_table()
        if table is None:
            return
        df = table.to_pandas()
        if "vector" in df.columns:
            df = df.drop(columns=["vector"])
        df.to_parquet(MEMORY_PARQUET_PATH, index=False)
        logger.debug("Parquet backup exported: %d rows", len(df))
    except Exception as e:
        logger.warning("Parquet export failed: %s", e)


# ── Migration from JSON ────────────────────────────────────────────

def migrate_json_to_lance():
    """One-time migration: reads memories_boss.json → LanceDB."""
    table = _get_table()
    if table is not None and table.count_rows() > 0:
        print(f"✅  LanceDB already has {table.count_rows()} memories. Skipping migration.")
        return

    if not os.path.exists(LEGACY_JSON):
        print(f"⚠️  No legacy JSON found at {LEGACY_JSON}. Nothing to migrate.")
        return

    try:
        with open(LEGACY_JSON, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        logger.error("Failed to read legacy JSON: %s", e)
        return

    if not json_data:
        return

    from embedding_model import embed_texts

    print(f"📦  Migrating {len(json_data)} memories from JSON → LanceDB...")

    contents = []
    records = []
    for mem_id, mem in json_data.items():
        content = mem.get("content", "")
        contents.append(content)
        records.append({
            "id": mem_id,
            "content": content,
            "tags": ",".join(mem.get("tags", [])),
            "metadata_str": json.dumps(mem.get("metadata", {})),
            "created_at": mem.get("created_at", datetime.now().isoformat()),
            "importance": 1.0,
        })

    vectors = embed_texts(contents)
    for i, rec in enumerate(records):
        rec["vector"] = vectors[i]

    _create_table_with_data(records)
    _export_parquet()

    print(f"✅  Migrated {len(records)} memories to LanceDB + Parquet backup.")
    logger.info("Migrated %d memories from JSON to LanceDB", len(records))


# ── Load memories (backwards-compatible wrapper) ────────────────────

def load_memories() -> dict:
    """Backwards-compatible: returns memories as a dict keyed by ID."""
    all_mems = get_all_memories()
    return {m["id"]: m for m in all_mems}


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"DB Path  : {MEMORY_DB_PATH}")
    print(f"Parquet  : {MEMORY_PARQUET_PATH}")
    print(f"Enabled  : {MEMORY_ENABLED}")
    total = count_memories()
    print(f"Total    : {total}")
    for m in get_all_memories()[:5]:
        print(f"  [{m['id']}] {m['content'][:70]}...")
    print("\n--- Semantic search: 'favorite color' ---")
    for r in semantic_search("what is my favorite color?", limit=3):
        print(f"  [{r['_similarity']:.3f}] {r['content'][:70]}...")
