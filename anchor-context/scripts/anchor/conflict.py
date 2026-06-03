"""Conflict detection and supersedes chain management.

Scheme 1 (Anchor Immutability): anchors are never modified, only superseded.
New anchors don't overwrite old ones — they link to them via the supersedes
chain, preserving the full evolution history for reconstruction.
"""

from .models import Anchor, AnchorSequence, EntityClass


def detect_conflicts(
    existing: AnchorSequence,
    incoming: AnchorSequence,
    overlap_threshold: float = 0.6,
) -> list[tuple[int, int]]:
    """Detect conflicts between existing and incoming anchors.

    A conflict occurs when two anchors share overlapping entities
    (same entity_class + high text similarity) but were created at
    different times. Returns list of (existing_anchor_index, incoming_anchor_index)
    pairs where the incoming anchor should supersede the existing one.

    Tie-breaking: when multiple existing anchors match one incoming anchor,
    the one with the highest overlap score wins. When tied, the newer (higher pos)
    version supersedes the older (lower pos).
    """
    conflicts = []

    for i, new_a in enumerate(incoming.anchors):
        best_score = 0.0
        best_existing_idx = -1

        for j, old_a in enumerate(existing.anchors):
            if old_a.is_superseded:
                continue

            score = _entity_overlap_score(old_a, new_a)

            # Bonus if same entity class
            if old_a.entity_class == new_a.entity_class:
                score += 0.2

            # Cap at 1.0
            score = min(score, 1.0)

            if score >= overlap_threshold and score > best_score:
                best_score = score
                best_existing_idx = j
            elif score == best_score and best_score >= overlap_threshold:
                # Tie: newer (higher pos) supersedes older
                if new_a.pos > old_a.pos:
                    best_existing_idx = j

        if best_existing_idx >= 0:
            conflicts.append((best_existing_idx, i))

    return conflicts


def mark_superseded(sequence: AnchorSequence, conflicts: list[tuple[int, int]]):
    """Apply supersedes relationships based on detected conflicts.

    Old anchors are marked as superseded and the new anchor records
    which anchors it supersedes.
    """
    for old_idx, new_idx in conflicts:
        if old_idx < len(sequence.anchors) and new_idx < len(sequence.anchors):
            old_anchor = sequence.anchors[old_idx]
            new_anchor = sequence.anchors[new_idx]

            # Mark old as superseded
            old_anchor.is_superseded = True

            # Record chain link in new anchor
            old_id = _anchor_id(old_anchor)
            if old_id not in new_anchor.supersedes:
                new_anchor.supersedes.append(old_id)


def _entity_overlap_score(a1: Anchor, a2: Anchor) -> float:
    """Compute entity text overlap score between two anchors.

    Uses Jaccard-like similarity on character trigrams for fuzzy matching.
    Returns 0.0-1.0.
    """
    if a1.entity == a2.entity:
        return 1.0

    # Trigram overlap
    tri1 = set(_trigrams(a1.entity.lower()))
    tri2 = set(_trigrams(a2.entity.lower()))

    if not tri1 or not tri2:
        return 0.0

    intersection = tri1 & tri2
    union = tri1 | tri2
    return len(intersection) / len(union)


def _trigrams(s: str) -> list[str]:
    """Generate character trigrams from a string."""
    s = "  " + s + " "
    return [s[i:i+3] for i in range(len(s) - 2)]


def _is_state_change(old: Anchor, new: Anchor) -> bool:
    """Check if two anchors represent a state change (same entity, different type)."""
    return (old.entity == new.entity and
            old.anchor_type != new.anchor_type)


def _anchor_id(anchor: Anchor) -> int:
    """Stable ID for an anchor."""
    raw = f"{anchor.entity}{anchor.pos}{anchor.anchor_type.value}"
    return hash(raw) & 0x7FFFFFFF
