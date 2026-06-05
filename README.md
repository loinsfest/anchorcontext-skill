# Never lose context in Claude Code again

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-622%2F0-brightgreen.svg)](tests/)
[![Release](https://img.shields.io/badge/release-v1.1.1-blue.svg)](https://github.com/loinsfest/anchorcontext-skill/releases)

> **Status:** Core pipeline complete, 622 unit tests passing (including 100-500 message ultra-long verification). **No production experience yet — feedback welcome.**

**Claude Code's automatic compaction makes it forget what you were working on.** Anchor Context extracts key decisions, bugs, and data points from your conversation before compaction, then injects them back so Claude remembers. One command. Zero cost.

```
Before: "What were we working on again?" — Claude after every compaction
After:  "Here are the key anchors from your previous session..."
         [DECISION] decided to use Redis SETNX
         [ANOMALY]  auth.ts:42 JWT race condition [ERR_005]
         [FACT]     PostgreSQL 14.2
```

## Quick Start

```bash
# Install
git clone https://github.com/loinsfest/anchorcontext-skill.git
cd anchorcontext-skill && bash install.sh  # or .\install.ps1 on Windows

# Done. Now just use Claude Code normally.
# When context fills up, anchors are saved automatically.
# Say "anchor context" to see them at any time.
```

## The Problem

Long Claude Code sessions lose context when compaction kicks in (~95% context usage). The summary Claude generates is lossy — decisions, bug discoveries, error codes, version numbers all get summarized away. You end up repeating yourself.

## Our Solution

Anchor Context saves **structured anchors** — verb-noun pairs with exact data — before compaction. After compaction, they're injected back into the session. You can also query them on demand.

```
Conversation (10000 tokens)
        │
        ▼ Extract (regex, zero LLM cost)
~300 tokens of anchors
        │
        ▼ Compact
        │
        ▼ Inject anchors back
Claude remembers what mattered
```

**93% compression. Zero API cost. Fully automatic.**

## How It Works

```
Message: "We decided to use Redis SETNX for distributed locking"
  → [DECISION] decided → Redis SETNX

Message: "Found JWT race condition at auth.ts line 42. Error code ERR_005."
  → [ANOMALY] race condition → auth.ts  [line:42, ERR_005]

Message: "Database will be PostgreSQL 14.2, pool size 20."
  → [FACT] PostgreSQL 14.2 [pool 20]
```

Each anchor is a **verb + noun + data values**. Verbs are classified: DECISION, DISCOVERY, ANOMALY, CONSTRAINT, FACT. Nouns carry exact numbers and tags.

When you query "what database are we using?", the system finds PostgreSQL via its semantic tags (`database`, `storage`, `SQL`) and returns the relevant window of anchors.

## Features

- **Fully automatic** — 3 Claude Code hooks (PreCompact, SessionStart, Stop) handle everything
- **93% compression** — 918 tokens → 75 tokens on a 30-message conversation
- **Query-aware** — ask about specific topics, get targeted context back
- **Zero-cost mode** — regex extraction, no API calls needed
- **LLM-enhanced mode** — optional DeepSeek integration for better selection and auto-tagging
- **Ultra-long verified** — tested at 100, 200, and 500 message scale

## Install

**Requirements:** Python 3.9+ (standard library only, no pip install needed)

### macOS / Linux / Git Bash
```bash
git clone https://github.com/loinsfest/anchorcontext-skill.git
cd anchorcontext-skill && bash install.sh
```

### Windows (PowerShell)
```powershell
git clone https://github.com/loinsfest/anchorcontext-skill.git
cd anchorcontext-skill
.\install.ps1
```

## Usage

1. **Automatic** — anchors save during compaction. Work normally.
2. **Say "anchor context"** — display saved anchors anytime.
3. **Ask specific questions** — "what was that bug in auth.ts?" — Claude uses anchors to answer.

```bash
# View saved anchors manually
python ~/.claude/skills/anchor-context/scripts/inject.py --format
```

### LLM-enhanced mode (optional)

```bash
export DEEPSEEK_API_KEY="your-key"
```

One $0.001 API call per compaction. Replaces all hand-crafted rules with LLM judgment — better entity selection, auto-generated semantic tags.

## Performance

| Metric | Value |
|--------|-------|
| Compression (30 msgs) | 93% |
| Compression (500 msgs) | 80%+ |
| Extraction speed (50 msgs) | <0.1s |
| Extraction speed (500 msgs) | <3s |
| Unit tests | 622 passing, 0 failures |

## Project Structure

```
anchorcontext-skill/
├── anchor-context/SKILL.md           # Skill definition
├── anchor-context/scripts/
│   ├── inject.py                     # SessionStart[compact] hook
│   ├── pre_compact.py                # PreCompact hook
│   ├── stop_backup.py                # Stop hook (backup)
│   └── anchor/
│       ├── models.py                 # VerbAnchor + NounAnchor + AnchorGraph
│       ├── extractor.py              # Bidirectional extraction pipeline
│       ├── verbs.py                  # 180+ verb lexicon (EN + CN)
│       ├── judge.py                  # LLM judge + zero-cost fallback
│       ├── reconstructor.py          # Hybrid retrieval (TF-IDF + FTS5)
│       ├── store.py / store_sqlite.py  # JSON + SQLite storage
│       └── ...
├── tests/ (622 tests)
├── hooks/hooks.json
└── install.sh / install.ps1
```

## vs Other Approaches

| Approach | Compression | API Cost | What It Preserves |
|----------|:-----------:|:--------:|-------------------|
| Claude Code default compaction | 70-80% | $0 | AI-summarized prose |
| claude-mem (76K stars) | 80-85% | High | SQLite + vector embeddings |
| Extractive keywords | 88-93% | $0 | Full sentences matching keywords |
| **Anchor Context** | **93%** | **$0** | **Structured verb-noun pairs + exact data** |

The difference: we don't compress text. We extract what matters and throw away the rest. When you need context back, we reconstruct it from the anchors.

## Limitations

| Limitation | Mitigation |
|------------|------------|
| No production experience yet | 622 tests, verified at 500-msg scale |
| Query matching is keyword-based | LLM mode adds semantic tags |
| Requires Python 3.9+ | Bundled skill, zero pip install |

## Keywords

`claude-code` `context-compression` `context-window` `compaction` `memory` `anchor` `skill` `llm` `deepseek` `claude` `ai-coding` `productivity` `open-source`

## Contact

Email: **2865157073@qq.com**

## License

MIT — see [LICENSE](LICENSE)
