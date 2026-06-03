#!/usr/bin/env python3
"""SessionStart[compact] hook handler — injects anchor context after compaction.

Claude Code fires SessionStart hooks with matcher "compact" after a
compaction completes. This script loads saved anchors and outputs them
as hookSpecificOutput.additionalContext for injection into the new session.

The output is JSON on stdout following the Claude Code hook output schema:
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "<formatted anchors>"
  }
}

Usage:
    python inject.py              # Loads anchors, outputs JSON for hook injection
    python inject.py --format     # Human-readable format (stdout display)
"""

import json
import os
import sys
from pathlib import Path

# Add bundled anchor/ module to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from anchor import AnchorSequence
from anchor.store import AnchorStore
from anchor.formatter import format_for_injection, format_verbose


def _escape_json(s: str) -> str:
    """Escape string for embedding in JSON value."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def handle_inject():
    """Load saved anchors and output as hook injection JSON."""
    store = AnchorStore()
    sequences = store.load_all_sequences()

    if not sequences:
        # No anchors saved — output empty (still valid JSON)
        _output_empty()
        return

    # Format anchors for injection
    context = format_for_injection(sequences)

    # Build hook output per Claude Code SessionStart schema
    escaped = _escape_json(context)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": escaped,
        }
    }

    print(json.dumps(output, ensure_ascii=False))

    n_total = sum(len(s.get_active()) for s in sequences)
    print(f"[anchor-context] Injected {n_total} anchors from {len(sequences)} sessions",
          file=sys.stderr)


def _output_empty():
    """Output valid empty injection JSON."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "",
        }
    }, ensure_ascii=False))


def handle_display():
    """Display saved anchors in human-readable format."""
    store = AnchorStore()
    sequences = store.load_all_sequences()

    if not sequences:
        print("(No saved anchors)")
        return

    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Session: {seq.session_id}")
        print(f"{'='*60}")
        active = seq.get_active()
        for i, anchor in enumerate(active):
            print(format_verbose(anchor, i))
        print(f"\n{len(active)} active anchors ({len(seq.anchors)} total)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--format":
        handle_display()
    else:
        handle_inject()
