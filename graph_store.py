"""
Graph Store Module — Neural Second Brain Foundation
===================================================
Phase 1 storage layer for graph memory (nodes + edges).
This module is intentionally dependency-light (sqlite3 only) and
backward-compatible with the current vector-first memory pipeline.
"""

import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# -- Config -----------------------------------------------------------
try:
    import config
    GRAPH_ENABLED = getattr(config, "GRAPH_ENABLED", True)
    GRAPH_DB_PATH = getattr(config, "GRAPH_DB_PATH", "Conversation/graph_memory.db")
except ImportError:
    GRAPH_ENABLED = True
    GRAPH_DB_PATH = "Conversation/graph_memory.db"

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DB_PATH = os.path.join(_BASE_DIR, GRAPH_DB_PATH)
_GRAPH_FOLDER = os.path.dirname(GRAPH_DB_PATH)


_conn: Optional[sqlite3.Connection] = None


def is_graph_enabled() -> bool:
    """Check whether graph memory is enabled."""
    return GRAPH_ENABLED


def _ensure_folder():
    """Ensure graph database folder exists."""
    os.makedirs(_GRAPH_FOLDER, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    """Return a lazy sqlite connection and initialize schema on first use."""
    global _conn
    if _conn is None:
        _ensure_folder()
        _conn = sqlite3.connect(GRAPH_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        init_graph_store()
        logger.info("Graph store connected: %s", GRAPH_DB_PATH)
    return _conn


def init_graph_store():
    """Create graph schema if missing."""
    conn = _get_conn() if _conn is None else _conn
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT DEFAULT 'memory',
            tags TEXT DEFAULT '',
            metadata_str TEXT DEFAULT '{}',
            created_at TEXT,
            importance REAL DEFAULT 1.0,
            updated_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship TEXT DEFAULT 'related',
            weight REAL DEFAULT 0.5,
            metadata_str TEXT DEFAULT '{}',
            created_at TEXT,
            UNIQUE(source_id, target_id, relationship)
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            source_id TEXT DEFAULT '',
            target_id TEXT DEFAULT '',
            query_text TEXT DEFAULT '',
            score REAL DEFAULT 0.0,
            metadata_str TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_events(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_id ON activity_events(id)")
    cur.execute(
        """
        INSERT OR IGNORE INTO graph_meta(key, value, updated_at)
        VALUES ('graph_version', '1', ?)
        """,
        (datetime.now().isoformat(),),
    )
    conn.commit()


def _bump_graph_version():
    """Increment graph version for lightweight UI real-time polling."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM graph_meta WHERE key = 'graph_version' LIMIT 1")
    row = cur.fetchone()
    current = 1
    if row and row[0] is not None:
        try:
            current = int(row[0])
        except Exception:
            current = 1
    new_version = current + 1
    cur.execute(
        """
        INSERT INTO graph_meta(key, value, updated_at)
        VALUES ('graph_version', ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (str(new_version), datetime.now().isoformat()),
    )
    conn.commit()


def get_graph_version() -> int:
    """Return monotonically increasing graph version."""
    if not is_graph_enabled():
        return 0
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM graph_meta WHERE key = 'graph_version' LIMIT 1")
    row = cur.fetchone()
    if not row:
        return 0
    try:
        return int(row[0])
    except Exception:
        return 0


def upsert_node(node_id: str, content: str,
                node_type: str = "memory",
                tags: Optional[list] = None,
                metadata: Optional[dict] = None,
                created_at: Optional[str] = None,
                importance: float = 1.0):
    """Insert or update a generic graph node."""
    if not is_graph_enabled():
        return

    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    tags_str = ",".join(tags or [])

    cur.execute(
        """
        INSERT INTO nodes(id, content, node_type, tags, metadata_str, created_at, importance, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            content=excluded.content,
            node_type=excluded.node_type,
            tags=excluded.tags,
            metadata_str=excluded.metadata_str,
            importance=excluded.importance,
            updated_at=excluded.updated_at
        """,
        (
            node_id,
            content,
            node_type,
            tags_str,
            json.dumps(metadata or {}),
            created_at or now,
            importance,
            now,
        ),
    )
    conn.commit()
    _bump_graph_version()


def upsert_memory_node(memory_id: str, content: str,
                       tags: Optional[list] = None,
                       metadata: Optional[dict] = None,
                       created_at: Optional[str] = None,
                       importance: float = 1.0):
    """Insert or update a memory node in graph storage."""
    upsert_node(
        node_id=memory_id,
        content=content,
        node_type="memory",
        tags=tags,
        metadata=metadata,
        created_at=created_at,
        importance=importance,
    )


def add_edge(source_id: str, target_id: str,
             relationship: str = "related",
             weight: float = 0.5,
             metadata: Optional[dict] = None):
    """Create or update a relationship edge between two nodes."""
    if not is_graph_enabled() or not source_id or not target_id:
        return
    if source_id == target_id:
        return

    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    cur.execute(
        """
        INSERT INTO edges(source_id, target_id, relationship, weight, metadata_str, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, target_id, relationship) DO UPDATE SET
            weight=excluded.weight,
            metadata_str=excluded.metadata_str
        """,
        (
            source_id,
            target_id,
            relationship,
            max(0.0, min(1.0, float(weight))),
            json.dumps(metadata or {}),
            now,
        ),
    )
    conn.commit()
    _bump_graph_version()


def delete_node(node_id: str):
    """Delete a node and all edges that reference it."""
    if not is_graph_enabled() or not node_id:
        return

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
    cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
    conn.commit()
    _bump_graph_version()


def list_nodes(limit: int = 100) -> list[dict]:
    """Return a list of graph nodes."""
    if not is_graph_enabled():
        return []

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, content, node_type, tags, metadata_str, created_at, importance, updated_at
        FROM nodes
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    out = []
    for row in cur.fetchall():
        out.append(
            {
                "id": row["id"],
                "content": row["content"],
                "node_type": row["node_type"],
                "tags": [t for t in (row["tags"] or "").split(",") if t],
                "metadata": json.loads(row["metadata_str"] or "{}"),
                "created_at": row["created_at"] or "",
                "importance": row["importance"] if row["importance"] is not None else 1.0,
                "updated_at": row["updated_at"] or "",
            }
        )
    return out


def get_node(node_id: str) -> Optional[dict]:
    """Get one node by ID."""
    if not is_graph_enabled() or not node_id:
        return None

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, content, node_type, tags, metadata_str, created_at, importance, updated_at
        FROM nodes
        WHERE id = ?
        LIMIT 1
        """,
        (node_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "content": row["content"],
        "node_type": row["node_type"],
        "tags": [t for t in (row["tags"] or "").split(",") if t],
        "metadata": json.loads(row["metadata_str"] or "{}"),
        "created_at": row["created_at"] or "",
        "importance": row["importance"] if row["importance"] is not None else 1.0,
        "updated_at": row["updated_at"] or "",
    }


def get_nodes_by_ids(node_ids: list[str]) -> dict[str, dict]:
    """Batch fetch nodes by IDs."""
    if not is_graph_enabled() or not node_ids:
        return {}

    unique_ids = [nid for nid in dict.fromkeys(node_ids) if nid]
    if not unique_ids:
        return {}

    placeholders = ",".join(["?"] * len(unique_ids))

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, content, node_type, tags, metadata_str, created_at, importance, updated_at
        FROM nodes
        WHERE id IN ({placeholders})
        """,
        tuple(unique_ids),
    )

    out = {}
    for row in cur.fetchall():
        out[row["id"]] = {
            "id": row["id"],
            "content": row["content"],
            "node_type": row["node_type"],
            "tags": [t for t in (row["tags"] or "").split(",") if t],
            "metadata": json.loads(row["metadata_str"] or "{}"),
            "created_at": row["created_at"] or "",
            "importance": row["importance"] if row["importance"] is not None else 1.0,
            "updated_at": row["updated_at"] or "",
        }
    return out


def list_edges(limit: int = 200, min_weight: float = 0.0) -> list[dict]:
    """Return a list of graph edges."""
    if not is_graph_enabled():
        return []

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source_id, target_id, relationship, weight, metadata_str, created_at
        FROM edges
        WHERE weight >= ?
        ORDER BY weight DESC, created_at DESC
        LIMIT ?
        """,
        (min_weight, limit),
    )

    out = []
    for row in cur.fetchall():
        out.append(
            {
                "source": row["source_id"],
                "target": row["target_id"],
                "relationship": row["relationship"],
                "weight": row["weight"] if row["weight"] is not None else 0.0,
                "metadata": json.loads(row["metadata_str"] or "{}"),
                "created_at": row["created_at"] or "",
            }
        )
    return out


def get_neighbors(node_id: str, limit: int = 20, min_weight: float = 0.0) -> list[dict]:
    """Get direct neighborhood for a node (phase-1 traversal helper)."""
    if not is_graph_enabled() or not node_id:
        return []

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source_id, target_id, relationship, weight, metadata_str, created_at
        FROM edges
        WHERE (source_id = ? OR target_id = ?) AND weight >= ?
        ORDER BY weight DESC, created_at DESC
        LIMIT ?
        """,
        (node_id, node_id, min_weight, limit),
    )

    neighbors = []
    for row in cur.fetchall():
        other = row["target_id"] if row["source_id"] == node_id else row["source_id"]
        neighbors.append(
            {
                "neighbor_id": other,
                "relationship": row["relationship"],
                "weight": row["weight"] if row["weight"] is not None else 0.0,
                "metadata": json.loads(row["metadata_str"] or "{}"),
                "created_at": row["created_at"] or "",
            }
        )
    return neighbors


def get_graph_snapshot(node_limit: int = 500, edge_limit: int = 1000) -> dict:
    """Return graph snapshot payload for future UI/API consumption."""
    return {
        "version": get_graph_version(),
        "nodes": list_nodes(limit=node_limit),
        "edges": list_edges(limit=edge_limit),
    }


def add_activity_event(event_type: str,
                       source_id: str = "",
                       target_id: str = "",
                       query_text: str = "",
                       score: float = 0.0,
                       metadata: Optional[dict] = None):
    """Append a lightweight runtime activity event for UI animation."""
    if not is_graph_enabled() or not event_type:
        return

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO activity_events(event_type, source_id, target_id, query_text, score, metadata_str, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            source_id or "",
            target_id or "",
            query_text or "",
            float(score or 0.0),
            json.dumps(metadata or {}),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()


def get_activity_events(since_id: int = 0, limit: int = 200) -> dict:
    """Fetch recent activity events after an event id for real-time UI polling."""
    if not is_graph_enabled():
        return {"events": [], "last_id": since_id}

    conn = _get_conn()
    cur = conn.cursor()
    safe_limit = max(1, min(int(limit or 200), 500))
    cur.execute(
        """
        SELECT id, event_type, source_id, target_id, query_text, score, metadata_str, created_at
        FROM activity_events
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (max(0, int(since_id or 0)), safe_limit),
    )

    events = []
    last_id = since_id
    for row in cur.fetchall():
        eid = int(row["id"])
        if eid > last_id:
            last_id = eid
        events.append(
            {
                "id": eid,
                "type": row["event_type"],
                "source": row["source_id"] or "",
                "target": row["target_id"] or "",
                "query": row["query_text"] or "",
                "score": float(row["score"] or 0.0),
                "metadata": json.loads(row["metadata_str"] or "{}"),
                "created_at": row["created_at"] or "",
            }
        )

    return {"events": events, "last_id": int(last_id or 0)}


if __name__ == "__main__":
    print(f"Graph enabled: {GRAPH_ENABLED}")
    print(f"Graph DB path: {GRAPH_DB_PATH}")
    init_graph_store()
    print("Graph store ready.")
