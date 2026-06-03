"""Anchor formatting for context injection and display."""

from .models import Anchor, AnchorSequence


def format_for_injection(sequences: list[AnchorSequence]) -> str:
    """Format anchors for hook-specific additionalContext injection.

    Produces a compact representation suitable for embedding in the
    system prompt. Superseded anchors are excluded.

    Uses positional indexing with PRIMARY marker for query-hit anchors
    to distinguish causal vs. temporal adjacency.
    """
    if not sequences:
        return ""

    lines = ["[锚点上下文 — Anchor Context]"]
    lines.append("以下是此前对话的关键锚点序列（按时间排列）：")
    lines.append("")

    for seq in sequences:
        active = seq.get_active()
        if not active:
            continue

        lines.append(f"## Session: {seq.session_id[:12]}")
        for i, anchor in enumerate(active):
            type_tag = anchor.anchor_type.value
            data_str = ""
            if anchor.data_values:
                data_str = f" [{', '.join(anchor.data_values)}]"

            # Use positional index for reference
            lines.append(f"  [{i}] [{type_tag}] {anchor.entity}{data_str}")

        lines.append("")

    lines.append("---")
    lines.append("使用方式：说「使用锚点上下文」查看完整锚点，或针对具体话题提问以触发位置检索重建。")

    return "\n".join(lines)


def format_compact(sequence: AnchorSequence, query_hit_index: int = -1) -> str:
    """Format a window of anchors around a query hit for reconstruction.

    The hit anchor gets a PRIMARY marker to distinguish it from temporally-
    adjacent neighbors. This prevents the LLM from confusing adjacency
    (just happened nearby in time) with causality (caused by / caused this).
    """
    active = sequence.get_active()
    if not active:
        return "(No active anchors)"

    lines = []
    for i, anchor in enumerate(active):
        type_tag = anchor.anchor_type.value
        marker = " ★ PRIMARY" if i == query_hit_index else ""
        data_str = ""
        if anchor.data_values:
            data_str = f" [{', '.join(anchor.data_values)}]"

        lines.append(f"[{i}]{marker} [{type_tag}] {anchor.entity}{data_str}")

    return "\n".join(lines)


def format_verbose(anchor: Anchor, index: int) -> str:
    """Full detail format for a single anchor (debugging/display)."""
    parts = [
        f"--- Anchor [{index}] ---",
        f"  Entity:      {anchor.entity}",
        f"  Type:        {anchor.anchor_type.value}",
        f"  Class:       {anchor.entity_class.value}",
        f"  Position:    {anchor.pos}",
    ]
    if anchor.data_values:
        parts.append(f"  Data:        {', '.join(anchor.data_values)}")
    if anchor.supersedes:
        parts.append(f"  Supersedes:  {anchor.supersedes}")
    if anchor.is_superseded:
        parts.append(f"  ⚠ SUPERSEDED")
    return "\n".join(parts)
