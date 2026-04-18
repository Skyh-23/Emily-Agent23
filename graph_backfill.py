"""
Graph Backfill — One-time population of graph from existing LanceDB memories.
Run this once after enabling the graph system for the first time.

Usage:
    python graph_backfill.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import logging
logging.basicConfig(level=logging.WARNING)

from graph_store import init_graph_store, list_nodes, list_edges
from graph_bridge import backfill_all_memories

print("=" * 55)
print("  Graph Backfill — LanceDB → SQLite Graph")
print("=" * 55)

# Show current graph state before
nodes_before = len(list_nodes(limit=99999))
edges_before = len(list_edges(limit=99999))
print(f"\nBefore: {nodes_before} nodes, {edges_before} edges")

# Run backfill
synced, failed = backfill_all_memories()

# Show result
nodes_after = len(list_nodes(limit=99999))
edges_after = len(list_edges(limit=99999))

print(f"\nAfter:  {nodes_after} nodes, {edges_after} edges")
print(f"\nResult: {synced} synced, {failed} failed")
print("\nOpen http://127.0.0.1:8010 after running graph_ui_server.py to see the graph.")
print("=" * 55)
