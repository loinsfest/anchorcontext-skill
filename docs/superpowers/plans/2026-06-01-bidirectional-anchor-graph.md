# Bidirectional Anchor Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace flat Anchor model with bidirectional VerbAnchor/NounAnchor graph. Each anchor links to exactly one nearest opposite-type neighbor. Top-N significance scoring. Target: 92% compression.

**Architecture:** Two symmetric anchor types, bipartite graph, rewritten extractor with unified scoring pipeline.

**Tech Stack:** Python 3.9+ (stdlib only — existing deps)

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `anchor-context/scripts/anchor/models.py` | MODIFY | Add VerbAnchor, NounAnchor, AnchorGraph |
| `anchor-context/scripts/anchor/extractor.py` | REWRITE | Pair verbs↔nouns, score, Top-N cap |
| `anchor-context/scripts/anchor/reconstructor.py` | MODIFY | Graph-based reconstruction with link traversal |
| `anchor-context/scripts/anchor/formatter.py` | MODIFY | Format both anchor types for injection |
| `anchor-context/scripts/anchor/store.py` | MODIFY | Serialize/deserialize AnchorGraph |
| `anchor-context/scripts/anchor/store_sqlite.py` | MODIFY | SQLite persistence for graph anchors |
| `tests/test_core.py` | MODIFY | Update tests for new models, add graph tests |

---

### Task 1: Add VerbAnchor, NounAnchor, AnchorGraph to models.py

**Files:** Modify `models.py`

**Goal:** Data models for the bidirectional graph.

- [ ] **Step 1: Write models**

```python
@dataclass
class VerbAnchor:
    """A key action extracted from conversation. Links to nearest noun."""
    id: str = ""
    entity: str = ""
    anchor_type: AnchorType = AnchorType.FACT
    pos: int = 0
    nearest_noun_id: str = ""        # Link to NounAnchor.id
    data_hints: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"v{hash(self.entity + str(self.pos)) & 0x7FFFFFFF:x}"


@dataclass
class NounAnchor:
    """A key entity extracted from conversation. Links to nearest verb."""
    id: str = ""
    entity: str = ""
    entity_class: EntityClass = EntityClass.TERM
    pos: int = 0
    nearest_verb_id: str = ""        # Link to VerbAnchor.id
    data_values: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"n{hash(self.entity + str(self.pos)) & 0x7FFFFFFF:x}"


@dataclass
class AnchorGraph:
    """Bipartite graph of verb and noun anchors from a conversation."""
    session_id: str = ""
    verb_anchors: list[VerbAnchor] = field(default_factory=list)
    noun_anchors: list[NounAnchor] = field(default_factory=list)

    @property
    def total_anchors(self) -> int:
        return len(self.verb_anchors) + len(self.noun_anchors)

    @property
    def total_chars(self) -> int:
        return (sum(len(v.entity) + len(v.anchor_type.value) + 5 for v in self.verb_anchors) +
                sum(len(n.entity) + len(n.entity_class.value) + 5 for n in self.noun_anchors))

    def find_noun(self, noun_id: str) -> Optional[NounAnchor]:
        for n in self.noun_anchors:
            if n.id == noun_id:
                return n
        return None

    def find_verb(self, verb_id: str) -> Optional[VerbAnchor]:
        for v in self.verb_anchors:
            if v.id == verb_id:
                return v
        return None
```

- [ ] **Step 2: Run existing tests** — `pytest tests/test_core.py -q` — expected: compilation error on old Anchor references (RED)

---

### Task 2: Rewrite extractor.py — bidirectional pairing

**Files:** Rewrite `extractor.py` `extract_anchors()` function

**Goal:** Extract VerbAnchors and NounAnchors with mutual nearest-neighbor links.

- [ ] **Step 1: Write `extract_graph()` function**

```python
def extract_graph(messages: list[dict], session_id: str = None) -> AnchorGraph:
    """Extract bidirectional verb-noun anchor graph from messages."""
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]

    # Concatenate messages with position tracking (same as before)
    full_text, msg_boundaries = _concat_messages(messages)
    if not full_text.strip():
        return AnchorGraph(session_id=session_id)

    verb_matches = segment_text(full_text)
    noun_matches = _extract_entities(full_text)

    # Phase 1: Create verb anchors — each verb links to nearest noun
    verb_anchors = []
    for verb_text, verb_type, v_start, v_end in verb_matches:
        if verb_type == "FACT" and verb_text.lower() in _STOP_ENTITIES:
            continue

        nearest_noun = _find_nearest(noun_matches, v_start, WINDOW_SIZE)
        if nearest_noun is None:
            continue

        noun_text, noun_class, n_start, n_end = nearest_noun
        if not _is_proper_entity(noun_text):
            continue

        data_hints = _extract_data_values(full_text[max(0,v_start-40):min(len(full_text),n_end+40)])
        v = VerbAnchor(
            entity=verb_text,
            anchor_type=AnchorType(verb_type),
            pos=v_start,
            data_hints=data_hints,
        )
        verb_anchors.append(v)

    # Phase 2: Create noun anchors — each noun links to nearest verb
    noun_anchors = []
    for noun_text, noun_class, n_start, n_end in noun_matches:
        if noun_class != EntityClass.DATA and not _is_proper_entity(noun_text):
            continue

        nearest_verb = _find_nearest_verb(verb_matches, n_start, WINDOW_SIZE)
        if nearest_verb is None:
            continue

        v_text, v_type, v_start, v_end = nearest_verb
        data_values = _extract_data_values(full_text[max(0,n_start-40):min(len(full_text),n_end+40)])
        n = NounAnchor(
            entity=noun_text,
            entity_class=noun_class,
            pos=n_start,
            data_values=data_values,
        )
        noun_anchors.append(n)

    # Phase 3: Resolve bidirectional links (nearest-noun in VerbAnchor -> NounAnchor.id)
    for v in verb_anchors:
        nearest = _find_nearest(noun_matches, v.pos, WINDOW_SIZE)
        if nearest:
            for n in noun_anchors:
                if n.pos == nearest[2]:
                    v.nearest_noun_id = n.id
                    break

    for n in noun_anchors:
        nearest = _find_nearest_verb(verb_matches, n.pos, WINDOW_SIZE)
        if nearest:
            for v in verb_anchors:
                if v.pos == nearest[2]:
                    n.nearest_verb_id = v.id
                    break

    # Phase 4: Score + Top-N cap
    candidates = _score_candidates(verb_anchors, noun_anchors, noun_matches)
    target = max(8, len(messages) // 2)
    keep_verbs, keep_nouns = _select_top_n(candidates, target)

    graph = AnchorGraph(session_id=session_id)
    graph.verb_anchors = keep_verbs
    graph.noun_anchors = keep_nouns
    return graph
```

- [ ] **Step 2: Helper functions**

```python
def _find_nearest(matches, pos, window):
    """Find nearest match to position within window. Returns (text, class, start, end) or None."""
    best, best_dist = None, float('inf')
    for item in matches:
        if len(item) == 4:
            text, cls, start, end = item
        else:
            text, cls, start, end = item[0], EntityClass.TERM, item[2], item[3]
        dist = abs(pos - start)
        if dist < best_dist and dist < window:
            best_dist = dist
            best = (text, cls, start, end)
    return best

def _find_nearest_verb(verb_matches, pos, window):
    """Find nearest verb to position within window."""
    best, best_dist = None, float('inf')
    for v_text, v_type, v_start, v_end in verb_matches:
        dist = abs(pos - v_start)
        if dist < best_dist and dist < window:
            best_dist = dist
            best = (v_text, v_type, v_start, v_end)
    return best

_VERB_SCORE = {"DECISION": 5, "DISCOVERY": 4, "ANOMALY": 3, "CONSTRAINT": 2, "FACT": 1}

def _score_candidates(verb_anchors, noun_anchors, noun_matches):
    """Score all verb-noun pairs. Returns sorted list of (score, 'verb'|'noun', index)."""
    pairs = []
    for i, v in enumerate(verb_anchors):
        specificity = 1.0
        if v.data_hints: specificity += 2.0
        if ' ' in v.entity: specificity += 0.5
        score = _VERB_SCORE.get(v.anchor_type.value, 1) * specificity
        pairs.append((score, 'verb', i))
    for i, n in enumerate(noun_anchors):
        specificity = 1.0
        if n.data_values: specificity += 2.0
        if any(c.isupper() for c in n.entity[1:]): specificity += 1.0
        score = ENTITY_WEIGHT.get(n.entity_class, 0.5) * 10 * specificity
        pairs.append((score, 'noun', i))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs

def _select_top_n(candidates, n):
    """Select top N candidates, keeping symmetric links intact."""
    verbs, nouns = [], []
    kept_verb_ids, kept_noun_ids = set(), set()
    for score, kind, idx in candidates:
        if len(verbs) + len(nouns) >= n:
            break
        if kind == 'verb':
            verbs.append(verb_anchors[idx])
            kept_verb_ids.add(verb_anchors[idx].id)
        else:
            nouns.append(noun_anchors[idx])
            kept_noun_ids.add(noun_anchors[idx].id)
    return verbs, nouns
```

- [ ] **Step 3: Keep `extract_anchors()` as legacy wrapper**

```python
def extract_anchors(messages, session_id=None):
    """Legacy wrapper — returns AnchorSequence for backward compatibility."""
    graph = extract_graph(messages, session_id)
    seq = AnchorSequence(session_id=graph.session_id)
    for v in graph.verb_anchors:
        seq.add(Anchor(entity=v.entity, anchor_type=v.anchor_type,
                        entity_class=EntityClass.TERM, pos=v.pos,
                        data_values=v.data_hints))
    for n in graph.noun_anchors:
        seq.add(Anchor(entity=n.entity, anchor_type=AnchorType.FACT,
                        entity_class=n.entity_class, pos=n.pos,
                        data_values=n.data_values))
    return seq
```

---

### Task 3: Update formatter.py for graph display

**Files:** Modify `formatter.py`

**Goal:** Format verb anchors and noun anchors for injection and display.

- [ ] **Step 1: Add `format_graph()` function**

```python
def format_graph(graph: AnchorGraph) -> str:
    lines = ["[Anchor Context]"]
    if graph.verb_anchors:
        lines.append(f"Actions ({len(graph.verb_anchors)}):")
        for v in graph.verb_anchors:
            noun = graph.find_noun(v.nearest_noun_id)
            noun_str = f" -> {noun.entity}" if noun else ""
            hints = f" [{', '.join(v.data_hints)}]" if v.data_hints else ""
            lines.append(f"  [{v.anchor_type.value}] {v.entity}{hints}{noun_str}")
    if graph.noun_anchors:
        lines.append(f"Entities ({len(graph.noun_anchors)}):")
        for n in graph.noun_anchors:
            verb = graph.find_verb(n.nearest_verb_id)
            verb_str = f" <- {verb.entity}" if verb else ""
            vals = f" [{', '.join(n.data_values)}]" if n.data_values else ""
            lines.append(f"  [{n.entity_class.value}] {n.entity}{vals}{verb_str}")
    return "\n".join(lines)
```

---

### Task 4: Update reconstructor for graph traversal

**Files:** Modify `reconstructor.py`

**Goal:** Reconstruction that follows links: hit anchor -> traverse to opposite type -> slice text window.

---

### Task 5: Update tests

**Files:** Modify `test_core.py`

**Goal:** Test bidirectional extraction, link integrity, compression ratio.

- [ ] **Step 1: Write failing test**

```python
class TestBidirectionalGraph:
    def test_graph_compression(self):
        """10 messages should produce 8-15 anchors total (verbs + nouns)."""
        messages = [
            {"content": "We decided to use Redis SETNX for distributed lock"},
            {"content": "Found JWT race condition at auth.ts line 42 ERR_005"},
            {"content": "Database PostgreSQL 14.2 PgBouncer pool 20"},
            {"content": "API latency dropped from 200ms to 80ms after deploy"},
            {"content": "Memory leak LRU overflow 2.1GB to 180MB"},
            {"content": "Must add TOTP 2FA GDPR tokens deletable 30 days"},
            {"content": "Load test 500 RPS p50 45ms p95 88ms zero errors"},
            {"content": "ERR_005 present 14 days 3 percent users affected"},
            {"content": "Next OAuth2 Google GitHub social login 8 points"},
            {"content": "Redis session architecture for new features"},
        ]
        from anchor.extractor import extract_graph
        graph = extract_graph(messages)
        total = graph.total_anchors
        assert 5 <= total <= 15, f"Expected 5-15 anchors, got {total}"
        assert len(graph.verb_anchors) > 0, "Should have verb anchors"
        assert len(graph.noun_anchors) > 0, "Should have noun anchors"

    def test_links_bidirectional(self):
        """Every linked verb should point to an existing noun and vice versa."""
        messages = [
            {"content": "We decided to use Redis SETNX for distributed lock"},
            {"content": "Found JWT race condition at auth.ts line 42"},
        ]
        from anchor.extractor import extract_graph
        graph = extract_graph(messages)
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                assert graph.find_noun(v.nearest_noun_id) is not None
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                assert graph.find_verb(n.nearest_verb_id) is not None
```
