"""
Graph Bridge — Fire-and-Forget Graph Sync
==========================================
The ONLY connector between LanceDB (primary memory) and SQLite graph (UI mirror).

Design rules:
  - Always runs in a background daemon thread → never blocks main pipeline
  - All exceptions caught silently → graph failure NEVER affects memory save
  - Never imported by memory_store.py directly → called from outside after save
  - Graph is a READ-ONLY mirror for visualization — never touches LLM context
"""

import logging
import threading

logger = logging.getLogger(__name__)


def sync_memory_to_graph_async(
    memory_id: str,
    content: str,
    tags: list,
    metadata: dict,
    created_at: str,
    importance: float = 1.0,
):
    """
    Sync a saved memory to the graph in a background thread.
    Called AFTER successful LanceDB save. Completely non-blocking.

    Does two things:
      1. Upsert the memory as a graph node
      2. Extract entities/turns and create edges (phase-2)
    """
    def _run():
        try:
            from graph_store import upsert_memory_node, add_edge
            from graph_extractor import extract_memory_graph

            # 1. Upsert memory as a node
            upsert_memory_node(
                memory_id=memory_id,
                content=content,
                tags=tags,
                metadata=metadata,
                created_at=created_at,
                importance=importance,
            )

            # 2. Extract entities/turn structure and create edges
            extracted = extract_memory_graph(
                memory_id=memory_id,
                content=content,
                tags=tags,
                metadata=metadata,
            )

            from graph_store import upsert_node, add_edge as _add_edge

            for node in extracted.get("nodes", []):
                upsert_node(
                    node_id=node["id"],
                    content=node["content"],
                    node_type=node.get("node_type", "entity"),
                    tags=node.get("tags", []),
                    metadata=node.get("metadata", {}),
                    importance=node.get("importance", 1.0),
                )

            for edge in extracted.get("edges", []):
                _add_edge(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    relationship=edge.get("relationship", "related"),
                    weight=edge.get("weight", 0.5),
                    metadata=edge.get("metadata", {}),
                )

            logger.debug("Graph synced: %s (%d nodes, %d edges)",
                         memory_id,
                         len(extracted.get("nodes", [])),
                         len(extracted.get("edges", [])))

        except Exception as e:
            # Silent — graph is a mirror, never let it break the core pipeline
            logger.debug("Graph sync skipped for %s: %s", memory_id, e)

    t = threading.Thread(target=_run, daemon=True, name=f"graph-sync-{memory_id[-6:]}")
    t.start()


def backfill_all_memories():
    """
    One-time backfill: sync all existing LanceDB memories to graph.
    Run from graph_backfill.py — NOT called during normal operation.

    Returns: (synced_count, failed_count)
    """
    try:
        from memory_store import get_all_memories
        memories = get_all_memories()
    except Exception as e:
        logger.error("Backfill: could not load memories: %s", e)
        return 0, 0

    if not memories:
        print("No memories found to backfill.")
        return 0, 0

    print(f"Backfilling {len(memories)} memories into graph...")
    synced = 0
    failed = 0

    try:
        from graph_store import upsert_memory_node, upsert_node, add_edge
        from graph_extractor import extract_memory_graph
    except Exception as e:
        logger.error("Backfill: could not import graph modules: %s", e)
        return 0, len(memories)

    for i, mem in enumerate(memories):
        try:
            memory_id  = mem["id"]
            content    = mem["content"]
            tags       = mem.get("tags", [])
            metadata   = mem.get("metadata", {})
            created_at = mem.get("created_at", "")
            importance = mem.get("importance", 1.0)

            upsert_memory_node(
                memory_id=memory_id,
                content=content,
                tags=tags,
                metadata=metadata,
                created_at=created_at,
                importance=importance,
            )

            extracted = extract_memory_graph(
                memory_id=memory_id,
                content=content,
                tags=tags,
                metadata=metadata,
            )

            for node in extracted.get("nodes", []):
                upsert_node(
                    node_id=node["id"],
                    content=node["content"],
                    node_type=node.get("node_type", "entity"),
                    tags=node.get("tags", []),
                    metadata=node.get("metadata", {}),
                    importance=node.get("importance", 1.0),
                )

            for edge in extracted.get("edges", []):
                add_edge(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    relationship=edge.get("relationship", "related"),
                    weight=edge.get("weight", 0.5),
                    metadata=edge.get("metadata", {}),
                )

            synced += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(memories):
                print(f"  [{i+1}/{len(memories)}] synced {memory_id[:12]}...")

        except Exception as e:
            failed += 1
            logger.warning("Backfill failed for %s: %s", mem.get("id", "?"), e)

    return synced, failed
