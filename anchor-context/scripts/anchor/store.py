"""Anchor persistence layer — JSON-based storage under ~/.claude/anchors/."""

import json
import os
from pathlib import Path
from typing import Optional

from .models import AnchorSequence

DEFAULT_STORE_DIR = os.path.expanduser("~/.claude/anchors")


class AnchorStore:
    """Manages AnchorSequence persistence to disk.

    Each conversation session gets its own JSON file.
    Supports save, load, list, and prune operations.
    """

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_.")
        return self.store_dir / f"{safe_id}.json"

    def save_sequence(self, sequence: AnchorSequence):
        """Persist an AnchorSequence to disk.

        Runs conflict detection before saving: any anchor in the existing
        sequence that overlaps with anchors in the new sequence gets
        superseded.
        """
        path = self._path_for(sequence.session_id)

        # Load existing sequence for conflict detection
        existing = self.load_sequence(sequence.session_id)
        if existing is not None:
            from .conflict import detect_conflicts, mark_superseded
            conflicts = detect_conflicts(existing, sequence)
            mark_superseded(existing, conflicts)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sequence.to_dict(), f, ensure_ascii=False, indent=2)

    def load_sequence(self, session_id: str) -> Optional[AnchorSequence]:
        """Load a single AnchorSequence by session ID.

        Returns None if the file doesn't exist.
        """
        path = self._path_for(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seq = AnchorSequence.from_dict(data)

        # Recompute superseded state from the supersedes chain
        all_ids = {id(a) for a in seq.anchors}
        superseded_ids: set[int] = set()
        for anchor in seq.anchors:
            for sid in anchor.supersedes:
                superseded_ids.add(sid)

        for anchor in seq.anchors:
            if id(anchor) in superseded_ids:
                anchor.is_superseded = True

        return seq

    def load_all_sequences(self) -> list[AnchorSequence]:
        """Load all saved anchor sequences.

        Returns list sorted by file modification time (newest first).
        """
        sequences = []
        paths = sorted(
            self.store_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for path in paths:
            session_id = path.stem
            try:
                seq = self.load_sequence(session_id)
                if seq is not None and seq.get_active():
                    sequences.append(seq)
            except (KeyError, TypeError, ValueError):
                # Skip corrupted or incompatible files
                continue

        return sequences

    def prune(self, sequence: AnchorSequence, max_anchors: int = 200):
        """Remove oldest superseded anchors if total exceeds max_anchors.

        Anchors marked is_superseded are removed first. If still over limit,
        the oldest non-superseded anchors are removed.
        """
        if len(sequence.anchors) <= max_anchors:
            return

        # Compute superseded IDs from the chain
        superseded_ids: set[int] = set()
        for anchor in sequence.anchors:
            for sid in anchor.supersedes:
                superseded_ids.add(sid)

        # Remove superseded first
        sequence.anchors = [a for a in sequence.anchors if id(a) not in superseded_ids]

        # If still over limit, trim oldest (lowest pos)
        if len(sequence.anchors) > max_anchors:
            sequence.anchors.sort(key=lambda a: a.pos)
            sequence.anchors = sequence.anchors[-max_anchors:]

        # Persist the pruned sequence
        self.save_sequence(sequence)


def id(anchor) -> int:
    """Stable identity hash for an anchor based on content, not memory address."""
    raw = f"{anchor.entity}{anchor.pos}{anchor.anchor_type.value}"
    return hash(raw) & 0x7FFFFFFF
