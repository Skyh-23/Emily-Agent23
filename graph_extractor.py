"""
Graph Extractor Module — Neural Second Brain Phase 2
====================================================
Builds typed graph nodes/edges from ingested text using deterministic
rule-based extraction. The flow is best-effort and non-blocking.
"""

import hashlib
import re
from typing import Optional

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "as", "by",
    "you", "your", "i", "me", "my", "we", "our", "he", "she", "they", "them",
    "user", "emily", "boss", "sir", "from", "into", "about", "then", "than",
}


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:42] or "item"


def _stable_hash(text: str, size: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:size]


def _entity_node_id(entity: str) -> str:
    return f"ent-{_slug(entity)}-{_stable_hash(entity.lower())}"


def _doc_node_id(text: str, source: str) -> str:
    seed = f"{source}:{text[:240]}"
    return f"doc-{_slug(source)}-{_stable_hash(seed, size=10)}"


def _turn_node_id(memory_id: str, role: str) -> str:
    return f"turn-{role}-{memory_id}"


def _extract_candidate_entities(text: str, max_entities: int = 10) -> list[str]:
    # Priority 1: quoted phrases
    quoted = re.findall(r'"([^"]{3,80})"', text)

    # Priority 2: title-case entities and acronyms
    title_case = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,8})\b", text)

    # Priority 3: significant tokens (fallback)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text)
    fallback = [w for w in words if w.lower() not in _STOPWORDS]

    seen = set()
    out = []

    for group in (quoted, title_case, fallback):
        for item in group:
            ent = item.strip()
            key = ent.lower()
            if key in seen:
                continue
            if len(ent) < 3:
                continue
            seen.add(key)
            out.append(ent)
            if len(out) >= max_entities:
                return out

    return out


def extract_memory_graph(memory_id: str, content: str,
                         tags: Optional[list] = None,
                         metadata: Optional[dict] = None,
                         entity_limit: int = 10) -> dict:
    """
    Build typed nodes/edges for one saved memory record.

    Returns:
        {
          "nodes": [{id, content, node_type, tags, metadata, importance}],
          "edges": [{source_id, target_id, relationship, weight, metadata}]
        }
    """
    tags = tags or []
    metadata = metadata or {}

    nodes = []
    edges = []

    entities = _extract_candidate_entities(content, max_entities=entity_limit)
    for ent in entities:
        ent_id = _entity_node_id(ent)
        nodes.append(
            {
                "id": ent_id,
                "content": ent,
                "node_type": "entity",
                "tags": ["entity"],
                "metadata": {"source": "phase2_rule_extractor", "kind": "entity"},
                "importance": 1.0,
            }
        )
        edges.append(
            {
                "source_id": memory_id,
                "target_id": ent_id,
                "relationship": "mentions",
                "weight": 0.7,
                "metadata": {"source": "phase2_rule_extractor"},
            }
        )

    # Conversation turn structure: User -> Assistant, both part_of memory.
    if content.startswith("User:") and "\nEmily:" in content:
        user_text, emily_text = content.split("\nEmily:", 1)
        user_text = user_text.replace("User:", "", 1).strip()
        emily_text = emily_text.strip()

        user_turn = _turn_node_id(memory_id, "user")
        assistant_turn = _turn_node_id(memory_id, "assistant")

        nodes.append(
            {
                "id": user_turn,
                "content": user_text,
                "node_type": "event",
                "tags": ["conversation", "turn", "user"],
                "metadata": {"source": "phase2_rule_extractor", "role": "user"},
                "importance": 1.0,
            }
        )
        nodes.append(
            {
                "id": assistant_turn,
                "content": emily_text,
                "node_type": "event",
                "tags": ["conversation", "turn", "assistant"],
                "metadata": {"source": "phase2_rule_extractor", "role": "assistant"},
                "importance": 1.0,
            }
        )

        edges.append(
            {
                "source_id": user_turn,
                "target_id": assistant_turn,
                "relationship": "follows",
                "weight": 1.0,
                "metadata": {"source": "phase2_rule_extractor"},
            }
        )
        edges.append(
            {
                "source_id": user_turn,
                "target_id": memory_id,
                "relationship": "part_of",
                "weight": 1.0,
                "metadata": {"source": "phase2_rule_extractor"},
            }
        )
        edges.append(
            {
                "source_id": assistant_turn,
                "target_id": memory_id,
                "relationship": "part_of",
                "weight": 1.0,
                "metadata": {"source": "phase2_rule_extractor"},
            }
        )

    return {"nodes": nodes, "edges": edges}


def extract_document_graph(text: str, source: str,
                           metadata: Optional[dict] = None,
                           entity_limit: int = 12) -> dict:
    """
    Build typed nodes/edges for a document-like ingestion payload.
    """
    metadata = metadata or {}
    doc_id = _doc_node_id(text=text, source=source)

    nodes = [
        {
            "id": doc_id,
            "content": text[:1200],
            "node_type": "document",
            "tags": ["document", "ingested"],
            "metadata": {"source": source, **metadata},
            "importance": 1.2,
        }
    ]
    edges = []

    entities = _extract_candidate_entities(text, max_entities=entity_limit)
    for ent in entities:
        ent_id = _entity_node_id(ent)
        nodes.append(
            {
                "id": ent_id,
                "content": ent,
                "node_type": "entity",
                "tags": ["entity"],
                "metadata": {"source": "phase2_rule_extractor", "kind": "entity"},
                "importance": 1.0,
            }
        )
        edges.append(
            {
                "source_id": doc_id,
                "target_id": ent_id,
                "relationship": "mentions",
                "weight": 0.65,
                "metadata": {"source": source},
            }
        )

    return {"root_id": doc_id, "nodes": nodes, "edges": edges}
