"""Constraint graph construction from anchor sequences.

Builds a directed graph where anchors are nodes and edges represent
dependency relationships derived from CONSTRAINT-type anchors and
temporal adjacency patterns.

Scheme 4 (Constraint Self-Correction): entity overlap + temporal proximity
create cross-validation during reconstruction — if two anchors share entities
and are close in time, they reinforce each other's accuracy.
"""

from collections import defaultdict
from .models import Anchor, AnchorType, AnchorSequence


def build_constraint_graph(sequence: AnchorSequence) -> dict:
    """Build a dependency graph from an anchor sequence.

    Returns:
        {
            "nodes": [{"id": idx, "entity": ..., "type": ...}],
            "edges": [{"from": src_idx, "to": dst_idx, "relation": "causes"|"constrains"}],
            "clusters": [[idx1, idx2], ...]   # Cross-validation clusters
        }
    """
    active = sequence.get_active()
    if len(active) < 2:
        return {"nodes": [], "edges": [], "clusters": []}

    anchor_ids = {id(a): i for i, a in enumerate(active)}

    nodes = [
        {
            "id": i,
            "entity": a.entity,
            "type": a.anchor_type.value,
            "pos": a.pos,
        }
        for i, a in enumerate(active)
    ]

    edges = []
    for i, a in enumerate(active):
        if a.anchor_type == AnchorType.CONSTRAINT:
            # Link CONSTRAINT anchors to nearest preceding anchor
            for j in range(i - 1, -1, -1):
                if active[j].anchor_type != AnchorType.CONSTRAINT:
                    edges.append({
                        "from": j,
                        "to": i,
                        "relation": "constrains",
                    })
                    break

        # Link temporally adjacent anchors of different types
        if i > 0:
            prev = active[i - 1]
            if (a.anchor_type != prev.anchor_type and
                (a.pos - prev.pos) < 500):  # Within 500 chars
                edges.append({
                    "from": i - 1,
                    "to": i,
                    "relation": "follows",
                })

    # Cross-validation clusters: anchors sharing entities + close in time
    clusters = _find_cross_validation_clusters(active)

    return {"nodes": nodes, "edges": edges, "clusters": clusters}


def _find_cross_validation_clusters(anchors: list[Anchor]) -> list[list[int]]:
    """Find clusters of anchors that cross-validate each other.

    Two anchors form a cross-validation pair if they share entity text
    fragments and are within a temporal proximity window.
    """
    clusters = []
    visited: set[int] = set()

    for i, a1 in enumerate(anchors):
        if i in visited:
            continue

        cluster = [i]
        visited.add(i)

        for j, a2 in enumerate(anchors):
            if j in visited:
                continue
            # Entity overlap check
            if _entities_overlap(a1, a2) and abs(a1.pos - a2.pos) < 1000:
                cluster.append(j)
                visited.add(j)

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def _entities_overlap(a1: Anchor, a2: Anchor) -> bool:
    """Check if two anchors share meaningful entity text fragments."""
    e1 = a1.entity.lower()
    e2 = a2.entity.lower()

    if e1 == e2:
        return True

    # Check substring overlap for composite entities
    parts1 = set(e1.replace(" + ", " ").split())
    parts2 = set(e2.replace(" + ", " ").split())

    return bool(parts1 & parts2)
