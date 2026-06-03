---
name: anchor-context
description: "Use when the user says '锚点上下文', 'anchor context', '查看锚点', '注入锚点', 'anchor inject', '上下文锚点', '之前的对话', '之前讨论的', '之前说的', mentions wanting to recall or recover information from earlier in a long conversation, asks about something discussed before, or needs context reconstruction after compaction — provides anchor-based context recovery from compacted or earlier conversation segments"
---

# Anchor Context — 锚点上下文

## Overview

Extract minimal structured anchors from conversations and reconstruct context on demand. Instead of reading full conversation summaries, query specific topics and get a targeted anchor window with the relevant conversation points.

**Core principle:** Anchor = entity (noun) + verb (label) + data values (exact numbers). ~97% compression rate, zero LLM cost for extraction.

## When to Use

Activate when the user:
- Says trigger phrases: `锚点上下文`, `anchor context`, `查看锚点`, `注入锚点`
- Asks about something discussed earlier: "之前讨论的...", "之前说的..."
- Wants to review what happened before: "之前那个方案是什么"
- After a `/compact` — check if anchors were saved and offer to show them

## How It Works

```
Compression (automatic, PreCompact hook):
  Long conversation → context limit approaches
  → Hook fires → extracts anchors → saves to ~/.claude/anchors/
  → Original discarded, anchors preserved

Recovery (on demand, you trigger):
  User says "锚点上下文" or asks about prior topic
  → Load saved anchors → format for display
  → Present anchor summary to user
```

## Workflow

### 1. Check for Saved Anchors

Run the display command to see what's available:

```bash
python "<skill-path>/scripts/inject.py" --format
```

Replace `<skill-path>` with `~/.claude/skills/anchor-context` or the actual path.

### 2. Present Anchors to User

If anchors exist, present them in a structured way:
- Show the session's anchor summary
- Highlight KEY anchors (DECISION / ANOMALY types)
- Ask which topic they want to explore

### 3. Reconstruct on Query

When the user asks about a specific topic, use:
```bash
python "<skill-path>/scripts/anchor/reconstructor.py" "<user-query>"
```

This returns a positional window of anchors around the most relevant match.

<HARD-GATE>
Do NOT fabricate or hallucinate conversation history. If no anchors are saved, tell the user: "No saved anchors — anchors are saved automatically during compaction. Keep working and they'll be generated when context fills up."
</HARD-GATE>

## Red Flags

| Thought | Reality |
|---------|---------|
| "I'll just summarize from memory" | Memory summaries lose detail. Use saved anchors. |
| "The user didn't say the exact trigger phrase" | If they ask about past conversation, check for anchors. |
| "Anchors aren't relevant to this query" | Let the user decide. Show what's available first. |

## Commands Quick Reference

```bash
# Show saved anchors (human-readable)
python ~/.claude/skills/anchor-context/scripts/inject.py --format

# Manually save anchors from current session (for testing)
python ~/.claude/skills/anchor-context/scripts/pre_compact.py save

# Run unit tests
python -m pytest ~/.claude/skills/anchor-context/scripts/../tests/
```

## Technical Details

See [REFERENCE.md](REFERENCE.md) for:
- Anchor format specification
- Entity class definitions and weights
- Position-based retrieval algorithm
- Hook integration details
- Troubleshooting
