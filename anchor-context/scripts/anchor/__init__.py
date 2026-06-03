"""Anchor-based context compression for AI agents.

Extracts minimal structured anchors from conversations and reconstructs
context on demand via hybrid retrieval (TF-IDF position + SQLite FTS5).
~97% compression rate, zero LLM cost for extraction.
"""

from .models import (Anchor, AnchorType, EntityClass, AnchorSequence, ENTITY_WEIGHT,
                      VerbAnchor, NounAnchor, AnchorGraph)

__all__ = [
    "Anchor", "AnchorType", "EntityClass", "AnchorSequence", "ENTITY_WEIGHT",
    "VerbAnchor", "NounAnchor", "AnchorGraph",
]
