#!/usr/bin/env python3
"""Stop hook handler — saves anchors on session exit.

When Claude Code exits, this hook persists any remaining anchors
that haven't been saved yet (catches the case where compaction
never happened but the session had valuable content).

Usage:
    python stop_backup.py    # Reads stdin, extracts anchors if needed
"""

import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from anchor.extractor import extract_anchors
from anchor.store import AnchorStore
from anchor.store_sqlite import SqliteStore


def _extract_messages(data: dict) -> list[dict]:
    """Extract message list from hook input data."""
    if "messages" in data and isinstance(data["messages"], list):
        return data["messages"]
    if "conversation" in data:
        conv = data["conversation"]
        if isinstance(conv, dict) and "messages" in conv:
            return conv["messages"]
    return []


def handle_stop():
    """Save anchors on session exit if not already saved."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return

    messages = _extract_messages(data)
    if len(messages) < 3:
        # Too short to be meaningful
        return

    session_id = data.get("session_id", "")
    if not session_id:
        import uuid
        session_id = uuid.uuid4().hex[:12]

    # Check if this session already has anchors (from PreCompact)
    store = AnchorStore()
    existing = store.load_sequence(session_id)
    if existing is not None and len(existing.get_active()) > 0:
        # Already saved by PreCompact — skip
        return

    # Extract and save
    sequence = extract_anchors(messages, session_id=session_id)
    if not sequence.anchors:
        return

    store.save_sequence(sequence)

    # Also save to SQLite if available
    try:
        sqlite = SqliteStore()
        sqlite.save_sequence(sequence)
    except Exception:
        pass

    n_active = len(sequence.get_active())
    print(f"[anchor-context] Stop hook: saved {n_active} anchors from session {session_id}",
          file=sys.stderr)


if __name__ == "__main__":
    handle_stop()
