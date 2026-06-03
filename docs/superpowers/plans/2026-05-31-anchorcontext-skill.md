# AnchorContext Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code skill that extracts minimal anchor points from long conversations (~97% compression) and reconstructs context on demand, publishable on GitHub.

**Architecture:** Skill-centric design following superpowers patterns. SKILL.md in `~/.claude/skills/anchor-context/` for auto-discovery. SessionStart hook injects anchor bootstrap. PreCompact hook saves anchors as side-effect. Python core library bundled inside skill directory (zero pip dependency). Install script handles skill placement + hook registration.

**Tech Stack:** Python 3.9+ (stdlib only), Bash (install), Claude Code SKILL.md format

**Key insight from research:** PreCompact hooks are fire-and-forget (no output schema). Anchor injection uses SessionStart with `"matcher": "compact"` instead — fires after compaction completes and injects context via `additionalContext`.

---

## File Structure

```
anchorcontext-skill/
├── README.md                         # 项目说明 + 安装指南
├── LICENSE                           # MIT
├── install.sh                        # macOS/Linux 安装脚本
├── install.ps1                       # Windows 安装脚本
├── anchor-context/                   # Skill 目录（复制到 ~/.claude/skills/）
│   ├── SKILL.md                      # 主 skill 定义（触发条件 + 工作流）
│   ├── REFERENCE.md                  # 技术参考（锚点格式、API 文档）
│   └── scripts/                      # 可执行脚本
│       ├── inject.py                 # 锚点注入（SessionStart hook 调用）
│       ├── pre_compact.py            # 锚点保存（PreCompact hook 调用）
│       └── anchor/                   # 内嵌 anchor-core（零外部依赖）
│           ├── __init__.py
│           ├── models.py             # Anchor + AnchorSequence + EntityClass
│           ├── extractor.py          # 名词驱动提取管线
│           ├── verbs.py              # 动词词表（150+ 中英双语）
│           ├── reconstructor.py      # 位置检索 + 窗口切片 + 重建 prompt
│           ├── store.py              # AnchorSequence JSON 持久化
│           ├── formatter.py          # 上下文注入格式化
│           ├── constraints.py        # 约束图构建
│           └── conflict.py           # 冲突检测 + supersedes 链
├── hooks/
│   └── hooks.json                    # PreCompact + SessionStart hook 定义
├── tests/
│   ├── test_core.py                  # 核心库单元测试
│   └── test_e2e_llm.py              # 真实 LLM 验证
└── .claude-plugin/
    └── plugin.json                   # Plugin manifest（GitHub 发布用）
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `anchorcontext-skill/README.md`
- Create: `anchorcontext-skill/LICENSE`
- Create: `anchorcontext-skill/.claude-plugin/plugin.json`
- Create: `anchorcontext-skill/anchor-context/SKILL.md` (skeleton)
- Create: `anchorcontext-skill/hooks/hooks.json`

**Goal:** Set up project skeleton with all directory structure and minimal files.

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p anchorcontext-skill/{anchor-context/scripts/anchor,hooks,tests,.claude-plugin,docs/superpowers/plans}
```

- [ ] **Step 2: Write plugin.json**

Create `.claude-plugin/plugin.json`:
```json
{
  "name": "anchor-context",
  "description": "Anchor-based context compression for Claude Code — extract minimal anchors from long conversations and reconstruct context on demand. ~97% compression rate with zero LLM cost.",
  "version": "1.0.0",
  "author": {
    "name": "AnchorContext Contributors"
  },
  "homepage": "https://github.com/anchorcontext/anchorcontext-skill",
  "repository": "https://github.com/anchorcontext/anchorcontext-skill",
  "license": "MIT",
  "keywords": [
    "context-compression",
    "anchor",
    "memory",
    "compaction",
    "long-conversation"
  ]
}
```

- [ ] **Step 3: Write hooks.json**

Create `hooks/hooks.json`:
```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python \"${CLAUDE_PLUGIN_ROOT}/../anchor-context/scripts/pre_compact.py\" save",
            "async": true
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "python \"${CLAUDE_PLUGIN_ROOT}/../anchor-context/scripts/inject.py\"",
            "async": false
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Write SKILL.md skeleton**

Create `anchor-context/SKILL.md`:
```markdown
---
name: anchor-context
description: "Use when the user says '锚点上下文', 'anchor context', '查看锚点', '注入锚点', 'anchor inject', '上下文锚点', mentions wanting to recall previous conversation context, or needs to recover information from earlier in a long session — provides anchor-based context reconstruction from compacted conversations"
---

# Anchor Context

## Overview
...
```

- [ ] **Step 5: Write LICENSE (MIT)**

- [ ] **Step 6: Commit**

---

### Task 2: Anchor Core Models

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/__init__.py`
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/models.py`

**Goal:** Define the core data model — Anchor, AnchorType, EntityClass, AnchorSequence.

- [ ] **Step 1: Write models.py**

Port from the original design with these classes:

```python
"""Core data model for anchor-based context compression."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnchorType(Enum):
    DECISION = "DECISION"
    DISCOVERY = "DISCOVERY"
    ANOMALY = "ANOMALY"
    CONSTRAINT = "CONSTRAINT"
    FACT = "FACT"


class EntityClass(Enum):
    DATA = "DATA"    # line numbers, error codes, versions — always anchor
    TECH = "TECH"    # filenames, identifiers, uppercase words
    TERM = "TERM"    # Chinese terms, domain concepts


# Weights drive EntityClass priority during extraction
ENTITY_WEIGHT = {
    EntityClass.DATA: 1.0,
    EntityClass.TECH: 0.7,
    EntityClass.TERM: 0.5,
}


@dataclass
class Anchor:
    """A single anchor point extracted from conversation."""
    entity: str                           # The noun/entity (primary payload)
    anchor_type: AnchorType               # Verb classification label
    entity_class: EntityClass             # Entity class for scoring
    pos: int                              # Position in overall sequence
    data_values: list[str] = field(default_factory=list)  # Attached exact values
    supersedes: list[int] = field(default_factory=list)   # IDs of anchors this replaces
    _is_superseded: bool = False          # Whether this anchor has been superseded

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
            supersedes=d.get("supersedes", []),
        )


@dataclass
class AnchorSequence:
    """Time-ordered sequence of anchors from a conversation."""
    session_id: str
    anchors: list[Anchor] = field(default_factory=list)

    def add(self, anchor: Anchor):
        self.anchors.append(anchor)
        self.anchors.sort(key=lambda a: a.pos)

    def get_window(self, center_index: int, radius: int = 2) -> list[Anchor]:
        """Get a positional window around center_index."""
        start = max(0, center_index - radius)
        end = min(len(self.anchors), center_index + radius + 1)
        return [a for a in self.anchors[start:end] if not a.is_superseded]

    def get_active(self) -> list[Anchor]:
        """Get all non-superseded anchors."""
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
```

- [ ] **Step 2: Write __init__.py**

```python
"""anchor-core: Anchor-based context compression for AI agents."""

from .models import Anchor, AnchorType, EntityClass, AnchorSequence, ENTITY_WEIGHT

__all__ = ["Anchor", "AnchorType", "EntityClass", "AnchorSequence", "ENTITY_WEIGHT"]
```

- [ ] **Step 3: Commit**

---

### Task 3: Verb Lexicon

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/verbs.py`

**Goal:** Cross-domain verb lexicon (150+ Chinese + English verbs) with efficient regex segmentation.

[Content continued — this file has the 150+ verb list and segment_text()]

- [ ] **Step 1: Port verbs.py from original design with combined regex optimization**

- [ ] **Step 2: Commit**

---

### Task 4: Anchor Extractor

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/extractor.py`

**Goal:** Noun-driven extraction pipeline — zero LLM cost, pure regex + NLP.

- [ ] **Step 1: Port extractor.py from original design**

Key function: `extract_anchors(messages: list[dict]) -> AnchorSequence`
- Phase 1: DATA entities → always anchor
- Phase 2: Verb-nearby entities (TECH/TERM)
- Phase 3: TECH/TERM clusters without verbs

- [ ] **Step 2: Commit**

---

### Task 5: Store, Formatter, Conflict

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/store.py`
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/formatter.py`
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/conflict.py`
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/constraints.py`

**Goal:** Persistence, formatting, conflict detection, constraint graph.

- [ ] **Step 1: Port store.py** — `save_sequence()`, `load_sequence()`, `load_all_sequences()`, `prune()`

- [ ] **Step 2: Port formatter.py** — `format_for_injection()`, `format_compact()`

- [ ] **Step 3: Port conflict.py** — `detect_conflicts()`, `mark_superseded()`

- [ ] **Step 4: Port constraints.py** — constraint graph builder

- [ ] **Step 5: Commit**

---

### Task 6: Reconstructor (Position-Based Retrieval)

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/anchor/reconstructor.py`

**Goal:** TF-IDF position retrieval with window slicing and PRIMARY marker.

- [ ] **Step 1: Port reconstructor.py** — `SequenceRetriever` class

- [ ] **Step 2: Commit**

---

### Task 7: Hook Scripts (pre_compact.py + inject.py)

**Files:**
- Create: `anchorcontext-skill/anchor-context/scripts/pre_compact.py`
- Create: `anchorcontext-skill/anchor-context/scripts/inject.py`

**Goal:** Hook handler scripts that integrate with Claude Code lifecycle.

- [ ] **Step 1: Write pre_compact.py**

```python
#!/usr/bin/env python3
"""PreCompact hook handler — saves anchor context before compaction.

Reads conversation from stdin, extracts anchors, saves to ~/.claude/anchors/<session>.json
PreCompact hooks are fire-and-forget: no output schema, side effects only.
"""
import json
import sys
import os
from pathlib import Path

# Add bundled anchor/ to path
sys.path.insert(0, str(Path(__file__).parent))

from anchor import extract_anchors, AnchorStore


def handle_save():
    """Extract anchors from stdin conversation and save to disk."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Empty or invalid input — nothing to save
        return

    messages = data.get("messages", [])
    session_id = data.get("session_id", "unknown")

    if not messages:
        return

    # Extract anchors from messages
    sequence = extract_anchors(messages)

    if not sequence.anchors:
        return

    # Save to ~/.claude/anchors/
    store = AnchorStore()
    store.save_sequence(sequence)

    # Log count for debugging
    store_dir = os.path.expanduser("~/.claude/anchors")
    n = len(sequence.get_active())
    print(f"[anchor-context] Saved {n} anchors from {len(messages)} messages to {store_dir}/{session_id}.json",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        handle_save()
```

- [ ] **Step 2: Write inject.py**

```python
#!/usr/bin/env python3
"""SessionStart[compact] hook handler — injects anchor context after compaction.

Reads saved anchors from ~/.claude/anchors/, formats them, and outputs
as hookSpecificOutput.additionalContext for injection into the session.
"""
import json
import sys
import os
from pathlib import Path

# Add bundled anchor/ to path
sys.path.insert(0, str(Path(__file__).parent))

from anchor import AnchorStore, format_for_injection


def handle_inject():
    """Load saved anchors and output as additionalContext."""
    store = AnchorStore()
    sequences = store.load_all_sequences()

    if not sequences:
        # No anchors saved — output empty (valid JSON still required)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": ""
            }
        }))
        return

    # Format anchors for injection
    context = format_for_injection(sequences)

    # Escape for JSON embedding
    escaped = context.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    output = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": escaped
        }
    })
    print(output)


if __name__ == "__main__":
    handle_inject()
```

- [ ] **Step 3: Commit**

---

### Task 8: SKILL.md (Complete)

**Files:**
- Modify: `anchorcontext-skill/anchor-context/SKILL.md`

**Goal:** Write the complete SKILL.md following superpowers patterns.

- [ ] **Step 1: Write complete SKILL.md**

Follow superpowers patterns:
- Description: ONLY triggering conditions, no workflow summary
- Overview + When to Use + Workflow + Commands
- Progressive disclosure (details → REFERENCE.md)
- HARD-GATE for critical decision points
- Red Flags table

- [ ] **Step 2: Write REFERENCE.md** — anchor format spec, API docs, troubleshooting

- [ ] **Step 3: Commit**

---

### Task 9: Install Scripts

**Files:**
- Create: `anchorcontext-skill/install.sh`
- Create: `anchorcontext-skill/install.ps1`

**Goal:** One-command install that copies skill to `~/.claude/skills/` and registers hooks.

- [ ] **Step 1: Write install.sh (macOS/Linux)**

5-step install:
1. Create `~/.claude/skills/anchor-context/` 
2. Copy skill files
3. Merge hooks into `~/.claude/settings.json` (using python for JSON manipulation)
4. Verify installation
5. Print usage instructions

- [ ] **Step 2: Write install.ps1 (Windows)**

Same logic in PowerShell.

- [ ] **Step 3: Commit**

---

### Task 10: Tests

**Files:**
- Create: `anchorcontext-skill/tests/test_core.py`
- Create: `anchorcontext-skill/tests/test_e2e_llm.py`

**Goal:** 39 unit tests + E2E LLM verification.

- [ ] **Step 1: Port test_core.py** — 39 unit tests from original

- [ ] **Step 2: Port test_e2e_llm.py** — Real LLM reconstruction test

- [ ] **Step 3: Run tests:** `python -m pytest tests/ -v`
  Expected: 39/39 pass

- [ ] **Step 4: Commit**

---

### Task 11: README.md + Polish

**Files:**
- Modify: `anchorcontext-skill/README.md`

**Goal:** Complete README with badges, architecture diagram, usage guide, install instructions.

- [ ] **Step 1: Write README.md** — Chinese + English, install guide, architecture, benchmarks

- [ ] **Step 2: Final verification** — Run install script on clean environment, verify skill discovery

- [ ] **Step 3: Commit**
