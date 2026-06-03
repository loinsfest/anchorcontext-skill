# Anchor Context — Technical Reference

## Anchor Format

```
[position] [TYPE] entity [+ attached data values]

Examples:
  [0] [DECISION] Redis SETNX  [distributed lock, auth.ts]
  [1] [DISCOVERY] JWT race condition  [line:42]
  [2] [CONSTRAINT] cross-pod sync  [must use distributed lock]
  [3] [FACT] PostgreSQL  [14.2]
```

### AnchorType Values

| Type | Meaning | Example Trigger Verbs |
|------|---------|----------------------|
| DECISION | A choice was made | 决定, 改用, chose, switched |
| DISCOVERY | Something was found | 发现, 定位, found, identified |
| ANOMALY | Error or unexpected behavior | 报错, 超时, error, crash |
| CONSTRAINT | Requirement or limitation | 必须, 不能, must, cannot |
| FACT | Neutral information | (fallback when no verb nearby) |

### EntityClass Values

| Class | Weight | Examples | Extraction Rule |
|-------|--------|----------|-----------------|
| DATA | 1.0 | `:42`, `ERR_001`, `14.2`, `500` | Always anchors |
| TECH | 0.7 | `auth.ts`, `Redis`, `SETNX` | Anchors if near a verb |
| TERM | 0.5 | `分布式锁`, `跨Pod部署` | Anchors in clusters |

## Extraction Pipeline

```
Messages → Phase 1 (DATA entities → always anchor)
        → Phase 2 (verb-nearby TECH/TERM entities)
        → Phase 3 (TECH/TERM clusters without verbs)
        → AnchorSequence (sorted by position)
```

Zero LLM calls. Pure regex + NLP with combined regex optimization (single-pass O(n) scan).

## Position-Based Retrieval

1. TF-IDF vectorize the query
2. Compute cosine similarity against each anchor entity
3. Find highest-scoring position
4. Slice window [hit - radius, hit + radius]
5. Mark hit with `★ PRIMARY` to distinguish from temporal neighbors

**Why not semantic search?** Temporal adjacency is often more informative than semantic similarity for conversation reconstruction. What happened near something is often relevant to understanding it.

## PRIMARY Marker

The query-hit anchor gets a `★ PRIMARY` marker in the reconstruction prompt:

```
[1] [DISCOVERY] JWT race condition  [line:42]
[2] ★ PRIMARY [DECISION] Redis SETNX  [auth.ts:42]
[3] [CONSTRAINT] cross-pod sync
```

This tells the LLM: "Anchor 2 is what the query matched. Anchors 1 and 3 are temporally adjacent — they may or may not be causally related."

Without PRIMARY: LLM might assume 1 caused 2 caused 3 (wrong).
With PRIMARY: LLM knows 2 is the focus, 1 and 3 provide context.

## Hook Integration

### PreCompact Hook
- **When:** Before context compaction
- **Behavior:** Fire-and-forget, saves anchors as side effect
- **Output:** None (async, no schema)
- **Script:** `pre_compact.py save`

### SessionStart[compact] Hook
- **When:** After compaction completes
- **Behavior:** Synchronous, injects anchors via `hookSpecificOutput.additionalContext`
- **Output:** JSON with `hookEventName` and `additionalContext`
- **Script:** `inject.py`

## Storage

Anchors are stored as JSON files under `~/.claude/anchors/`:

```
~/.claude/anchors/
├── abc123def456.json    # Most recent session
├── 789012345678.json    # Previous session
└── ...
```

### JSON Format

```json
{
  "session_id": "abc123def456",
  "anchors": [
    {
      "entity": "Redis SETNX",
      "anchor_type": "DECISION",
      "entity_class": "TECH",
      "pos": 142,
      "data_values": ["distributed lock"],
      "supersedes": []
    }
  ]
}
```

## Troubleshooting

### "No saved anchors"
- Anchors are saved during compaction (PreCompact hook). Work until context fills up.
- For testing: run `python pre_compact.py save` manually with sample JSON input.

### "python not found"
- Requires Python 3.9+. Install from https://python.org
- On Windows: use `python` not `python3`

### "Module anchor not found"
- The skill bundles anchor-core inside `scripts/anchor/`
- Ensure the skill is at `~/.claude/skills/anchor-context/`
- Run from the correct directory
