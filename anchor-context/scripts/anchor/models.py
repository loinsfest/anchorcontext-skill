"""Core data model for anchor-based context compression.

Anchor = entity (noun) + anchor_type (verb label) + data_values (exact numbers).
AnchorSequence = time-ordered list of anchors with positional window slicing.
"""

from dataclasses import dataclass, field
from enum import Enum


class AnchorType(Enum):
    """Verb classification — type label on the anchor."""
    DECISION = "DECISION"        # Chose Redis over Memcached
    DISCOVERY = "DISCOVERY"      # Found race condition at line 42
    ANOMALY = "ANOMALY"          # Timeout on /api/users
    CONSTRAINT = "CONSTRAINT"    # Must use dist lock across pods
    FACT = "FACT"                # PostgreSQL 14.2


class EntityClass(Enum):
    """Entity classification drives extraction priority.

    DATA entities (line numbers, error codes, versions) are always anchored
    because they're objective and query-relevant. TECH (filenames, identifiers)
    and TERM (Chinese concepts) have lower priority.
    """
    DATA = "DATA"    # Line numbers, error codes, version strings — always anchor
    TECH = "TECH"    # Filenames, identifiers, uppercase words
    TERM = "TERM"    # Chinese terms, domain concepts


ENTITY_WEIGHT = {
    EntityClass.DATA: 1.0,
    EntityClass.TECH: 0.7,
    EntityClass.TERM: 0.5,
}


@dataclass
class Anchor:
    """A single anchor point extracted from conversation.

    The entity (noun) is the primary payload. anchor_type (verb) is a label
    for classification. data_values carry exact numbers attached to the anchor.
    tags carry semantic category tags for search matching (NOT stored in entity text
    to keep compression high — tags are used during TF-IDF/FTS5 retrieval only).
    """
    entity: str
    anchor_type: AnchorType
    entity_class: EntityClass
    pos: int                                # Position in overall sequence (sort key)
    data_values: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # Semantic tags for search
    supersedes: list[int] = field(default_factory=list)   # IDs of replaced anchors
    _is_superseded: bool = False

    @property
    def is_superseded(self) -> bool:
        return self._is_superseded

    @is_superseded.setter
    def is_superseded(self, value: bool):
        self._is_superseded = value

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "anchor_type": self.anchor_type.value,
            "entity_class": self.entity_class.value,
            "pos": self.pos,
            "data_values": self.data_values,
            "tags": self.tags,
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        return cls(
            entity=d["entity"],
            anchor_type=AnchorType(d["anchor_type"]),
            entity_class=EntityClass(d["entity_class"]),
            pos=d["pos"],
            data_values=d.get("data_values", []),
            tags=d.get("tags", []),
            supersedes=d.get("supersedes", []),
        )


@dataclass
class AnchorSequence:
    """Time-ordered sequence of anchors from a single conversation session.

    Position-based retrieval: find index by TF-IDF, slice window around it.
    """
    session_id: str
    anchors: list[Anchor] = field(default_factory=list)

    def add(self, anchor: Anchor):
        self.anchors.append(anchor)
        # Maintain position order — critical for window slicing
        self.anchors.sort(key=lambda a: a.pos)

    def get_window(self, center_index: int, radius: int = 2) -> list[Anchor]:
        """Slice a positional window around center_index.

        Returns only active (non-superseded) anchors within the window.
        The center anchor is the query hit; neighbors provide temporal context.
        """
        start = max(0, center_index - radius)
        end = min(len(self.anchors), center_index + radius + 1)
        return [a for a in self.anchors[start:end] if not a.is_superseded]

    def get_active(self) -> list[Anchor]:
        return [a for a in self.anchors if not a.is_superseded]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "anchors": [a.to_dict() for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorSequence":
        seq = cls(session_id=d["session_id"])
        seq.anchors = [Anchor.from_dict(a) for a in d.get("anchors", [])]
        return seq


# ── Bidirectional Anchor Graph (v2) ────────────────────────────────────
# VerbAnchors and NounAnchors form a bipartite graph.
# Each verb links to its nearest noun. Each noun links to its nearest verb.
# This enables ~92% compression: only words + links, no context structure.

@dataclass
class VerbAnchor:
    """A key action (verb) from the conversation. Links to nearest noun."""
    entity: str
    anchor_type: AnchorType = AnchorType.FACT
    pos: int = 0
    nearest_noun_id: str = ""
    data_hints: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"v{hash(self.entity + str(self.pos)) & 0x7FFFFFFF:x}"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "entity": self.entity,
            "type": self.anchor_type.value, "pos": self.pos,
            "nearest_noun_id": self.nearest_noun_id,
            "data_hints": self.data_hints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VerbAnchor":
        return cls(entity=d["entity"], anchor_type=AnchorType(d["type"]),
                   pos=d["pos"], nearest_noun_id=d.get("nearest_noun_id", ""),
                   data_hints=d.get("data_hints", []))


@dataclass
class NounAnchor:
    """A key entity (noun) from the conversation. Links to nearest verb."""
    entity: str
    entity_class: EntityClass = EntityClass.TERM
    pos: int = 0
    nearest_verb_id: str = ""
    data_values: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # Semantic tags for search

    @property
    def id(self) -> str:
        return f"n{hash(self.entity + str(self.pos)) & 0x7FFFFFFF:x}"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "entity": self.entity,
            "class": self.entity_class.value, "pos": self.pos,
            "nearest_verb_id": self.nearest_verb_id,
            "data_values": self.data_values,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NounAnchor":
        return cls(entity=d["entity"], entity_class=EntityClass(d["class"]),
                   pos=d["pos"], nearest_verb_id=d.get("nearest_verb_id", ""),
                   data_values=d.get("data_values", []),
                   tags=d.get("tags", []))


@dataclass
class AnchorGraph:
    """Bipartite graph of verb and noun anchors from one conversation."""
    session_id: str = ""
    verb_anchors: list = field(default_factory=list)
    noun_anchors: list = field(default_factory=list)

    @property
    def total_anchors(self) -> int:
        return len(self.verb_anchors) + len(self.noun_anchors)

    @property
    def total_chars(self) -> int:
        return (sum(len(v.entity) + 10 for v in self.verb_anchors) +
                sum(len(n.entity) + 10 for n in self.noun_anchors))

    def find_noun(self, noun_id: str):
        for n in self.noun_anchors:
            if n.id == noun_id:
                return n
        return None

    def find_verb(self, verb_id: str):
        for v in self.verb_anchors:
            if v.id == verb_id:
                return v
        return None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "verb_anchors": [v.to_dict() for v in self.verb_anchors],
            "noun_anchors": [n.to_dict() for n in self.noun_anchors],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorGraph":
        g = cls(session_id=d["session_id"])
        g.verb_anchors = [VerbAnchor.from_dict(v) for v in d.get("verb_anchors", [])]
        g.noun_anchors = [NounAnchor.from_dict(n) for n in d.get("noun_anchors", [])]
        return g
