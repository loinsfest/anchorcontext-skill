# Bidirectional Anchor Graph — Design Spec

**Date:** 2026-06-01
**Status:** Approved

## 1. Core Idea

Traditional compression keeps context relationships and attention structure — expensive.
Anchor compression keeps only WORDS — cheap. Two types of anchors, minimal text, linked bidirectionally.

```
Traditional: "We decided to use Redis SETNX for distributed lock, database PostgreSQL 14.2"
             -> ~120 chars, preserves full sentence structure

Anchor Graph: 
  [DECISION] decided -> Redis SETNX
  [DATA]     PostgreSQL 14.2 [pool 20] -> will be
             -> ~60 chars, only words + links
```

Reconstruction rebuilds context on demand by following links and slicing original text windows.

## 2. Data Model

### VerbAnchor
```
entity: str          # verb text: "decided", "found", "crashed"
verb_type: enum      # DECISION | DISCOVERY | ANOMALY | CONSTRAINT
pos: int             # position in original text
nearest_noun_id: str # link to nearest NounAnchor (NULL if none)
data_hints: [str]    # numbers/error codes in window: ["line:42", "ERR_005"]
```

### NounAnchor
```
entity: str          # noun text: "Redis SETNX", "PostgreSQL 14.2", "auth.ts"
noun_class: enum     # DATA | TECH | TERM
pos: int             # position in original text
nearest_verb_id: str # link to nearest VerbAnchor (NULL if none)
data_values: [str]   # attached exact values: ["14.2", "pool 20"]
```

### Link invariant
- Each VerbAnchor links to exactly ONE NounAnchor (the nearest one in original text)
- Each NounAnchor links to exactly ONE VerbAnchor (the nearest one in original text)
- Links form a bipartite graph — no verb-verb or noun-noun edges

## 3. Extraction Algorithm

### Step 1: Find all candidates
```
verbs = segment_text(full_text)    # [(verb, type, start, end), ...]
nouns = extract_entities(full_text) # [(entity, class, start, end), ...]
```

### Step 2: Pair each verb with nearest noun
```
for each verb:
    find nearest noun within 80-char window
    create VerbAnchor(verb, type, pos, nearest_noun_id, data_hints)
```

### Step 3: Pair each noun with nearest verb  
```
for each noun:
    find nearest verb within 80-char window
    create NounAnchor(noun, class, pos, nearest_verb_id, data_values)
```

### Step 4: Filter noise
```
for each VerbAnchor:
    skip if verb is a stop word
    skip if verb has no linked noun (isolated)

for each NounAnchor:
    skip if noun is a common word (blacklist + proper noun check)
    skip if noun has no linked verb (isolated)
```

### Step 5: Score and cap
```
for each anchor:
    score = verb_type_weight * noun_class_weight * specificity_bonus

sort by score descending
keep top N: N = max(8, message_count // 2)
```

Verb type weights: DECISION=5, DISCOVERY=4, ANOMALY=3, CONSTRAINT=2
Noun class weights: DATA=1.0, TECH=0.7, TERM=0.5
Specificity bonus: has_digit=+2, multi_word=+1, pascal_case=+1

## 4. Storage Format

```json
{
  "session_id": "abc123",
  "verb_anchors": [
    {"id": "v1", "entity": "decided", "type": "DECISION", "pos": 14,
     "nearest_noun_id": "n1", "data_hints": []}
  ],
  "noun_anchors": [
    {"id": "n1", "entity": "Redis SETNX", "class": "TECH", "pos": 25,
     "nearest_verb_id": "v1", "data_values": []}
  ]
}
```

## 5. Reconstruction

```
query "Redis locking"
-> TF-IDF match finds noun "Redis SETNX" (n1)
-> Follow link: n1 -> v1 ("decided")
-> Slice text window around [v1.pos, n1.pos]
-> If more context needed, follow v1 links to other nouns near it
-> Build prompt: "Based on this text window, reconstruct the conversation context"
```

## 6. Compression Target

| Metric | Target |
|--------|--------|
| Anchors per message | ~0.4 (1 anchor per 2.5 messages) |
| Chars per anchor | ~25 (entity + type + link overhead) |
| 30 messages, 918 tokens | ~12 anchors, ~300 chars, ~75 tokens → **92% compression** |
| 50 messages, 2500 tokens | ~20 anchors, ~500 chars, ~125 tokens → **95% compression** |

## 7. Edge Cases

- **No verb nearby**: Noun anchor gets nearest_verb_id=NULL, excluded unless it has data_values
- **No noun nearby**: Verb anchor gets nearest_noun_id=NULL, excluded
- **Multiple nouns at same distance**: Pick the one with higher ENTITY_WEIGHT
- **Chinese text**: Chinese verb lexicon + Chinese noun patterns (already supported)
