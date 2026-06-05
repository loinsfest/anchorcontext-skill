#!/usr/bin/env python3
"""PreCompact hook handler — saves anchor context before compaction.

Key behaviors:
  1. Extracts anchors from conversation and saves to disk (side effect)
  2. Outputs compact instructions to stdout (exit code 0) telling the
     summarizer to preserve key anchors in the compressed summary

Claude Code PreCompact hook behavior:
  - Exit code 0: stdout becomes custom compact instructions for the summarizer
  - Exit code 2: blocks compaction entirely
  - stderr: shown to user, compaction proceeds normally

The anchors are later injected back via SessionStart[compact] hook.

Usage:
    python pre_compact.py save    # Reads stdin JSON, extracts and saves anchors
"""

import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from anchor import AnchorSequence
from anchor.extractor import extract_anchors
from anchor.store import AnchorStore


def handle_save():
    """Extract anchors from stdin conversation and persist to disk.

    Also outputs compact instructions (stdout) to help the summarizer
    preserve key information during compaction.
    """
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return

    messages = _extract_messages(data)
    if not messages:
        return

    session_id = data.get("session_id", "")
    if not session_id:
        import uuid
        session_id = uuid.uuid4().hex[:12]

    # Extract anchors
    sequence = extract_anchors(messages, session_id=session_id)
    if not sequence.anchors:
        return

    # Persist (JSON + SQLite)
    store = AnchorStore()
    store.save_sequence(sequence)

    try:
        from anchor.store_sqlite import SqliteStore
        sqlite = SqliteStore()
        sqlite.save_sequence(sequence)
    except Exception:
        pass

    n_active = len(sequence.get_active())
    store_dir = os.path.expanduser("~/.claude/anchors")
    print(f"[anchor-context] Saved {n_active} anchors from {len(messages)} messages",
          file=sys.stderr)

    # Output compact instructions (exit code 0 → stdout becomes custom compact instructions)
    _output_compact_instructions(sequence)


def _output_compact_instructions(sequence: AnchorSequence):
    """Output hints for the compaction summarizer about what to preserve.

    These instructions are passed to the LLM performing the compaction
    to guide what key information should survive in the summary.
    """
    active = sequence.get_active()
    if not active:
        return

    # Group by type for concise instructions
    decisions = [a for a in active if a.anchor_type.value == "DECISION"]
    anomalies = [a for a in active if a.anchor_type.value == "ANOMALY"]
    constraints = [a for a in active if a.anchor_type.value == "CONSTRAINT"]
    discoveries = [a for a in active if a.anchor_type.value == "DISCOVERY"]

    lines = [
        "[Anchor Context — compaction instructions]",
        "The following key decisions and findings should be preserved in your summary:",
    ]

    if decisions:
        lines.append(f"\nDecisions made ({len(decisions)}):")
        for a in decisions[:5]:
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            lines.append(f"  - {a.entity}{dv}")

    if discoveries:
        lines.append(f"\nDiscoveries ({len(discoveries)}):")
        for a in discoveries[:5]:
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            lines.append(f"  - {a.entity}{dv}")

    if anomalies:
        lines.append(f"\nAnomalies/Errors ({len(anomalies)}):")
        for a in anomalies[:5]:
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            lines.append(f"  - {a.entity}{dv}")

    if constraints:
        lines.append(f"\nConstraints ({len(constraints)}):")
        for a in constraints[:5]:
            lines.append(f"  - {a.entity}")

    lines.append(f"\nTotal: {len(active)} anchor points saved. Full anchor context can be")
    lines.append("retrieved by saying 'anchor context' or '锚点上下文' after compaction.")

    # Write to stdout — becomes custom compact instructions
    print("\n".join(lines))


def _extract_messages(data: dict) -> list[dict]:
    """Extract message list from PreCompact hook input data.

    Claude Code PreCompact hook stdin has NO messages field — it provides
    transcript_path pointing to a .jsonl file. Messages must be read from there.

    Also handles the legacy format (direct messages in stdin) for testing.
    """
    # Legacy: direct messages field (used by unit tests and manual invocation)
    if "messages" in data and isinstance(data["messages"], list):
        return data["messages"]

    # Legacy: nested in conversation
    if "conversation" in data:
        conv = data["conversation"]
        if isinstance(conv, dict) and "messages" in conv:
            return conv["messages"]

    # Production: read transcript_path from PreCompact hook
    transcript_path = data.get("transcript_path", "")
    if transcript_path and os.path.isfile(transcript_path):
        return _read_transcript(transcript_path)

    return []


def _read_transcript(transcript_path: str) -> list[dict]:
    """Read messages from a Claude Code .jsonl transcript file.

    Each line is a JSON object with: role, content (string or content block array).
    Extracts only text content for anchor extraction — skips tool_use, tool_result.
    """
    messages = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = obj.get("content", "")
                text = _extract_text_content(content)
                if text.strip():
                    messages.append({"content": text})
    except (OSError, IOError) as e:
        print(f"[anchor-context] Warning: cannot read transcript: {e}",
              file=sys.stderr)

    return messages


def _extract_text_content(content) -> str:
    """Extract plain text from a message's content field.

    Handles both:
      - Plain string: "hello world"
      - Content block array: [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)

    return str(content) if content else ""


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        handle_save()
    else:
        print("Usage: python pre_compact.py save", file=sys.stderr)
        print("  Reads conversation JSON from stdin, extracts anchors, saves to disk.",
              file=sys.stderr)
        sys.exit(1)
