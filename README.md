# Anchor Context

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-622%2F0-brightgreen.svg)](tests/)

> **Status:** Core pipeline complete, 622 unit tests passing (including 100-500 message ultra-long verification). **No production experience yet — feedback welcome.**

**Anchor-based context compression for Claude Code.** Extract minimal structured anchors from long conversations and reconstruct context on demand — query-aware, not summary-locked.

```
Traditional:  conversation → summary → store summary → read summary (one-shot, locked in)
Anchor:       conversation → extract anchors → store anchors → reconstruct on demand (query-aware)

10000 token conversation → ~300 token anchors → LLM reconstructs any topic on demand
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Conversation                          │
│  "We decided to use Redis SETNX for distributed lock."  │
│  "Found JWT race condition at auth.ts:42"               │
│  "Must sync across pods — need distributed lock"         │
│  "Database is PostgreSQL 14.2"                           │
└────────────────────────┬────────────────────────────────┘
                         │ Extract (zero LLM cost)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Anchor Sequence                        │
│  [0] [DECISION] Redis SETNX                             │
│  [1] [DISCOVERY] JWT race condition  [line:42]          │
│  [2] [CONSTRAINT] cross-pod sync                        │
│  [3] [FACT] PostgreSQL  [14.2]                          │
└────────────────────────┬────────────────────────────────┘
                         │ Store (~300 tokens, ~97% compression)
                         ▼
              ~/.claude/anchors/session.json

                         │ User asks: "What's the Redis lock approach?"
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Position-Based Retrieval                    │
│  TF-IDF → find position → slice window [1,2,3]          │
│  PRIMARY marker distinguishes hit from neighbors         │
└────────────────────────┬────────────────────────────────┘
                         │ Reconstruct
                         ▼
               LLM answers from anchor window
```

## Architecture

### Why anchors instead of summaries?

| Approach | Detail Preservation | Query Flexibility | Storage Cost |
|----------|-------------------|-------------------|--------------|
| Full conversation | 100% | High | 10000 tokens |
| Summary | ~30% | Low (locked in) | ~500 tokens |
| **Anchor Context** | **~80% reconstructable** | **High (query-aware)** | **~300 tokens** |

### Key Design Decisions

- **Noun-driven extraction**: Entities are primary payload, verbs are classification labels. DATA entities (line numbers, error codes, version strings) always anchor.
- **Position-based retrieval**: TF-IDF finds position in sequence, temporal window provides context. Not semantic search — temporal adjacency is often more informative.
- **Anchor immutability**: Old anchors are never modified, only superseded. Full evolution chains preserved for reconstruction.
- **PRIMARY marker**: Distinguishes the query-hit anchor from temporally-adjacent neighbors, preventing LLM from confusing adjacency with causality.
- **Zero LLM extraction cost**: Pure regex + NLP. No API calls needed for extraction.

## Quick Install

### macOS / Linux / Git Bash
```bash
git clone https://github.com/anchorcontext/anchorcontext-skill.git
cd anchorcontext-skill
bash install.sh
```

### Windows (PowerShell)
```powershell
git clone https://github.com/anchorcontext/anchorcontext-skill.git
cd anchorcontext-skill
.\install.ps1
```

**Requirements:** Python 3.9+ (standard library only, no pip install needed)

## Usage

1. **Automatic**: Anchors are saved during Claude Code compaction (PreCompact hook). Just work normally until context fills up.
2. **Manual trigger**: Say `anchor context` in any Claude Code session — the skill loads and displays saved anchors.
3. **Query reconstruction**: Ask a specific question about a prior topic — Claude uses the anchors to reconstruct relevant context.

```bash
# View saved anchors
python ~/.claude/skills/anchor-context/scripts/inject.py --format
```

## Project Structure

```
anchorcontext-skill/
├── anchor-context/                 # Skill directory
│   ├── SKILL.md                    # Skill definition (trigger conditions)
│   ├── REFERENCE.md                # Technical reference
│   └── scripts/
│       ├── inject.py               # SessionStart[compact] hook handler
│       ├── pre_compact.py          # PreCompact hook handler
│       ├── stop_backup.py          # Stop hook handler
│       └── anchor/                 # Anchor-core library (zero deps)
│           ├── models.py           # Anchor / VerbAnchor / NounAnchor / AnchorGraph
│           ├── extractor.py        # Bidirectional extraction pipeline
│           ├── verbs.py            # 180+ verb lexicon
│           ├── judge.py            # LLM significance judge + fallback
│           ├── reconstructor.py    # Hybrid retrieval (TF-IDF + FTS5)
│           ├── store.py            # JSON persistence
│           ├── store_sqlite.py     # SQLite + FTS5 backend
│           ├── formatter.py        # Context injection formatting
│           ├── conflict.py         # Conflict detection
│           └── constraints.py      # Constraint graph
├── hooks/hooks.json                # PreCompact + SessionStart + Stop hooks
├── tests/
│   ├── test_core.py                # 622 unit tests (59 classes)
│   ├── test_ultra_long.py          # Ultra-long stress tests (100-500 msgs)
│   ├── test_e2e_llm.py            # E2E LLM verification
│   └── data/                       # Generated test conversations
├── install.sh / install.ps1        # One-command install
├── .claude-plugin/plugin.json      # Plugin manifest
├── RELEASE-NOTES.md
└── README.md
```

## Verification

```bash
# Run unit tests
python -m pytest tests/ -v

# E2E test with real LLM
export ANCHOR_TEST_API_KEY="your-deepseek-key"
python tests/test_e2e_llm.py

# Dry-run (print prompts only)
python tests/test_e2e_llm.py --dry-run
```

## Performance

| Metric | Value |
|--------|-------|
| Compression rate | 93% (30-msg), 80%+ (500-msg) |
| Extraction speed | <0.1s for 50 messages, <3s for 500 |
| Unit tests | 622 passing, 0 failures |
| Ultra-long verified | 100/200/500 message conversations |

## Limitations

| Limitation | Mitigation |
|------------|------------|
| Chinese synonym zero-recall | SQLite FTS5 fallback in hybrid retrieval |
| Pure numbers without units | Known — tracked for improvement |
| Requires Python 3.9+ | Bundled skill, zero pip install |

## Ecosystem Comparison

Researched 8 popular Claude Code context/memory projects (May 2026):

| Project | Stars | Approach | LLM Cost | Retrieval | Storage | Our Differentiator |
|---------|-------|---------|-----------|-----------|---------|-------------------|
| **Anchor Context** | — | Regex anchors | **Zero** | TF-IDF + FTS5 | JSON + SQLite | Only zero-cost solution |
| claude-mem | 76K | AI compression | High | FTS5 + ChromaDB | SQLite + Vector | Cross-platform |
| CoMeT-CC | — | TLS proxy | Medium | 3-tier tree | Proxy cache | Lossless raw preservation |
| LCM | — | DAG summaries | Medium | FTS5 | SQLite | Promoted long-term memory |
| agentmemory | 46K | AI compression | High | Keyword+Vector+Graph | Local DB | 95% token reduction |
| Contexa | — | Git-branch model | Medium | Git-style log | File-based | K=1 optimal context |
| MemoryForge | — | Multi-hook | Zero | File search | STATE.md files | Session structure maps |
| claude-baton | — | SQLite checkpoints | Zero | SQL queries | SQLite | Git diff tracking |

**What we do differently (and why):**
1. **Zero LLM extraction cost** — All others use AI for compression. Our regex pipeline is free and deterministic.
2. **Anchor immutability** — Old anchors are superseded, never modified. Evolution chains preserved for reconstruction.
3. **PRIMARY marker** — Distinguishes query-hit from temporal neighbors. No other project addresses causal vs. temporal confusion.
4. **Position-based retrieval** — Temporal adjacency > semantic similarity for conversation reconstruction. Others use pure semantic search.
5. **Self-contained** — Bundled Python library. Zero pip install, zero external services.

## Contributing

Areas where contributions are especially valuable:
- BGE-M3 embedding integration for pure semantic fallback
- Multi-language verb lexicon expansion (JP, KR)
- CI/CD pipeline for cross-platform hook testing
- MCP tool wrappers for anchor management

## Contact

Email: **2865157073@qq.com**

## License

MIT — see [LICENSE](LICENSE)
