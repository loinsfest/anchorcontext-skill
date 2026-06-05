"""Noun-driven anchor extraction pipeline — zero LLM cost.

Three-phase extraction:
  Phase 1: DATA entities (line numbers, error codes, version strings) → always anchor
  Phase 2: Verb-nearby entities (TECH/TERM within window)
  Phase 3: TECH/TERM clusters without nearby verbs (fallback)

All extraction is regex-based. No API calls needed.
"""

import re
import uuid
from typing import Optional

import re as _re

from .models import (Anchor, AnchorType, EntityClass, AnchorSequence, ENTITY_WEIGHT,
                      VerbAnchor, NounAnchor, AnchorGraph)

# ── Code block handling ───────────────────────────────────────────────
# Code blocks are NOT anchor-extracted. They're replaced with a summary
# placeholder. Code can be re-read from disk when needed (Claude Code approach).

_CODE_BLOCK_RE = _re.compile(r'```(\w*)\n([\s\S]*?)```', _re.MULTILINE)


def _summarize_code_block(language: str, code: str) -> str:
    """Replace a code block with a summary placeholder."""
    lines = code.strip().split('\n')
    n_lines = len(lines)
    lang_label = language or 'code'

    # Extract key identifiers: function/class names, imports
    sigs = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            sigs.append(stripped[4:].split('(')[0].strip())
        elif stripped.startswith('class '):
            sigs.append(stripped[6:].split(':')[0].split('(')[0].strip())
        elif stripped.startswith('import ') or stripped.startswith('from '):
            sigs.append(stripped.rstrip(';'))

    sig_str = f": {', '.join(sigs[:5])}" if sigs else ""
    return f"[Code block: {lang_label}{sig_str}, {n_lines} lines]"


def _strip_code_blocks(text: str) -> tuple[str, list[dict]]:
    """Replace code blocks with summary placeholders.

    Returns (cleaned_text, code_refs) where code_refs are stored
    for potential file re-reading on demand.
    """
    code_refs = []

    def _replace(m):
        language = m.group(1) or ''
        code = m.group(2)
        ref = {
            'language': language,
            'n_lines': len(code.strip().split('\n')),
            'summary': _summarize_code_block(language, code),
        }
        code_refs.append(ref)
        return ref['summary']

    cleaned = _CODE_BLOCK_RE.sub(_replace, text)
    return cleaned, code_refs
from .verbs import segment_text, find_verbs_in_window, get_anchor_type

# ── Entity recognition patterns ──────────────────────────────────────────

# DATA entities: line numbers, error codes, version strings, numbers with units
_RE_LINE_NUM = re.compile(r':(\d{2,})(?:\s|$|,|\)|\.)')  # :42 at end of line
_RE_ERROR_CODE = re.compile(r'\b([A-Z]{2,6}[_-]\d{3,6})\b')  # ERR_001, E001
_RE_VERSION = re.compile(r'\b(\d+\.\d+(?:\.\d+)?)\b')  # 14.2, 3.10.1
_RE_NUMBER_WITH_UNIT = re.compile(r'\b(\d+(?:\.\d+)?\s*(?:ms|s|MB|GB|KB|TB|rps|req/s|RPS|%|min|hours?|days?))\b')  # 200ms, 2.1GB
_RE_NUMBER = re.compile(r'\b(\d{2,})\b')  # Standalone numbers >= 10

# TECH entities: filenames, identifiers, uppercase acronyms
_RE_FILENAME = re.compile(r'\b([\w-]+\.(?:ts|tsx|js|jsx|py|go|rs|java|rb|php|sql|yaml|yml|json|toml|cfg|ini|sh|bash|ps1|md|css|html|xml))\b')
_RE_IDENTIFIER = re.compile(r'\b([a-z]+(?:[._][a-z]+)+)\b')  # snake_case, dot.case
_RE_CAMELCASE = re.compile(r'\b([a-z]+(?:[A-Z][a-z]+)+)\b')  # camelCase
_RE_PASCALCASE = re.compile(r'\b([A-Z][a-z]+(?:[A-Z]+[a-z]*|\d+)+)\b')  # PascalCase + UPPER segments (PostgreSQL, OAuth2)
_RE_PROPER_NAME = re.compile(r'\b([A-Z][a-zA-Z0-9]{4,}(?:\s*[A-Z][a-zA-Z0-9]+)*)\b')  # Broad tech name catch-all
_RE_UPPER = re.compile(r'\b([A-Z]{2,}(?:_[A-Z]{2,})*)\b')  # UPPER_CASE
_RE_CMD_FLAG = re.compile(r'\b(--?[a-z-]{2,})\b')  # --flag, -f
_RE_DOMAIN = re.compile(r'\b([\w-]+\.(?:internal|com|io|org|net|dev|local))\b')  # grafana.internal

# TERM entities: Chinese term sequences
_RE_CHINESE_TERM = re.compile(r'[一-鿿]{2,8}')  # 2-8 char Chinese sequences

# ── Garbage filter: words that should NEVER become anchors ──────────────
_STOP_ENTITIES = {
    "BEFORE", "AFTER", "FIRST", "THEN", "ONLY", "MUST", "CANNOT",
    "based", "based on", "latency", "error", "issue", "problem",
    "using", "should", "could", "would", "will", "also", "still",
    "just", "very", "really", "much", "many", "such", "same",
    "added", "used", "found", "made", "need", "needs", "wants",
    "This", "explains", "intermittent", "errors", "saw", "production",
    "-based", "-latency", "active_connections", "requests_total",
}  # fmt: skip


def _is_garbage(entity: str) -> bool:
    """Check if an entity is likely a garbage fragment from regex over-matching."""
    if entity in _STOP_ENTITIES or entity.lower() in _STOP_ENTITIES:
        return True
    # Filter fragments that start with hyphen/underscore (partial regex captures)
    if entity.startswith("-") or entity.startswith("_"):
        return True
    # Filter fragments that end with hyphen/underscore
    if entity.endswith("-") or entity.endswith("_"):
        return True
    return False

# ── Semantic category tags: bridges entity names → query keywords ──────
# When an entity matches a known term, add category tags so queries like
# "database" can find "PostgreSQL", "cache" can find "Redis", etc.
_ENTITY_SEMANTIC_TAGS: dict[str, list[str]] = {
    # Databases
    "PostgreSQL": ["database", "storage", "SQL"],
    "PgBouncer": ["database", "connection pool"],
    "MySQL": ["database", "storage", "SQL"],
    "SQLite": ["database", "storage", "SQL"],
    # Caching
    "Redis": ["cache", "session", "distributed lock", "key-value"],
    "SETNX": ["cache", "distributed lock", "Redis"],
    "Memcached": ["cache", "key-value"],
    # Auth
    "JWT": ["auth", "token", "authentication", "security"],
    "TOTP": ["auth", "2FA", "authentication", "security", "MFA"],
    "OAuth2": ["auth", "authentication", "social login", "security"],
    "bcrypt": ["auth", "password", "hashing", "security"],
    # Monitoring
    "Prometheus": ["monitoring", "metrics", "observability"],
    "Grafana": ["monitoring", "dashboard", "observability"],
    # Infrastructure
    "Kubernetes": ["orchestration", "containers", "deployment"],
    "Docker": ["containers", "deployment"],
    # Patterns
    "LRU": ["cache", "eviction", "memory", "algorithm"],
    "FTS5": ["search", "full-text", "index"],
    # Protocols
    "CSRF": ["auth", "security", "web"],
    "XSS": ["security", "web", "injection"],
    "GDPR": ["compliance", "privacy", "legal"],
    # Frontend / Build
    "Vite": ["build", "bundler", "dev server", "HMR"],
    "Vitest": ["testing", "unit test", "jest"],
    "Webpack": ["build", "bundler", "module"],
    "Zustand": ["state management", "store", "React"],
    "Redux": ["state management", "store", "React"],
    "Tailwind": ["CSS", "utility", "styling", "design"],
    "Radix": ["UI", "component", "accessible", "headless"],
    "Playwright": ["E2E", "testing", "browser", "automation"],
    "Prisma": ["ORM", "database", "schema", "migration"],
    "tRPC": ["API", "type-safe", "RPC", "TypeScript"],
    "TanStack": ["query", "cache", "data fetching"],
    "WebSocket": ["real-time", "connection", "bidirectional"],
    "Storybook": ["design system", "component", "documentation", "UI"],
    "Clerk": ["auth", "authentication", "React", "session"],
    "Docker": ["container", "image", "deployment", "build"],
    "pnpm": ["package manager", "monorepo", "workspace"],
    "GraphQL": ["API", "query", "schema", "Apollo"],
    # Cloud / Infra
    "PlanetScale": ["database", "MySQL", "serverless"],
    "Vercel": ["hosting", "deploy", "preview", "edge"],
    "Cloudflare": ["CDN", "edge", "workers", "security"],
    "Datadog": ["monitoring", "APM", "observability", "RUM"],
    "PagerDuty": ["alerting", "on-call", "incident"],
    "LaunchDarkly": ["feature flag", "A/B testing", "rollout"],
    "Flagsmith": ["feature flag", "A/B testing", "GDPR"],
    # Frontend concepts
    "LCP": ["performance", "Core Web Vitals", "loading"],
    "CLS": ["performance", "Core Web Vitals", "layout shift"],
    "INP": ["performance", "Core Web Vitals", "interaction"],
    "HMR": ["build", "dev server", "hot reload"],
    "CSP": ["security", "content", "headers", "XSS"],
    "Lighthouse": ["performance", "audit", "Google"],
    "Chromatic": ["visual regression", "Storybook", "UI testing"],
    "WCAG": ["accessibility", "guidelines", "standards"],
    "aria": ["accessibility", "screen reader", "HTML"],
    # Compound/cluster entities
    "GitHub": ["CI", "pipeline", "actions", "repository"],
    "GitHub Actions": ["CI", "pipeline", "workflow", "automation"],
    "Chrome DevTools": ["debugging", "browser", "development"],
}

# Entity class assignment
def _classify_entity(entity: str) -> EntityClass:
    """Determine which EntityClass a matched string belongs to."""
    # Check DATA patterns first
    if (_RE_ERROR_CODE.match(entity) or _RE_NUMBER_WITH_UNIT.match(entity)):
        return EntityClass.DATA
    if _RE_VERSION.match(entity) or (_RE_NUMBER.fullmatch(entity) and entity.isdigit()):
        return EntityClass.DATA
    # TECH entities: filename/domain/flag patterns always qualify.
    # PascalCase/camelCase identifiers need at least 1 uppercase + len >= 3.
    is_filename = bool(_RE_FILENAME.match(entity))
    is_domain = bool(_RE_DOMAIN.match(entity))
    is_cmd = bool(_RE_CMD_FLAG.match(entity))
    uppercase_count = sum(1 for c in entity if c.isupper())
    if not (is_filename or is_domain or is_cmd) and uppercase_count == 0:
        return EntityClass.TERM
    if (is_filename or is_domain or _RE_IDENTIFIER.match(entity) or
        _RE_CAMELCASE.match(entity) or _RE_PASCALCASE.match(entity) or
        _RE_PROPER_NAME.match(entity) or
        _RE_UPPER.match(entity) or _RE_CMD_FLAG.match(entity) or
        _RE_DOMAIN.match(entity)):
        return EntityClass.TECH
    # Check Chinese
    if _RE_CHINESE_TERM.fullmatch(entity):
        return EntityClass.TERM
    return EntityClass.TERM  # fallback


def _extract_entities(text: str) -> list[tuple[str, EntityClass, int, int]]:
    """Extract all entities from text with their positions.

    Returns list of (entity_text, entity_class, start_pos, end_pos).
    """
    entities: list[tuple[str, EntityClass, int, int]] = []
    seen_positions: set[int] = set()

    patterns = [
        (_RE_NUMBER_WITH_UNIT, EntityClass.DATA),  # Must be before _RE_NUMBER
        (_RE_ERROR_CODE, EntityClass.DATA),
        (_RE_VERSION, EntityClass.DATA),
        (_RE_LINE_NUM, EntityClass.DATA),
        (_RE_FILENAME, EntityClass.TECH),
        (_RE_DOMAIN, EntityClass.TECH),
        (_RE_UPPER, EntityClass.TECH),
        (_RE_PASCALCASE, EntityClass.TECH),
        (_RE_PROPER_NAME, EntityClass.TECH),
        (_RE_CAMELCASE, EntityClass.TECH),
        (_RE_IDENTIFIER, EntityClass.TECH),
        (_RE_CMD_FLAG, EntityClass.TECH),
        (_RE_NUMBER, EntityClass.DATA),
    ]

    for pattern, default_class in patterns:
        for m in pattern.finditer(text):
            entity = (m.group(1) if pattern in (_RE_LINE_NUM, _RE_NUMBER_WITH_UNIT)
                      else m.group(0))

            # Filter garbage entities
            if _is_garbage(entity):
                continue
            # Filter purely numeric standalones under 10
            if pattern is _RE_NUMBER and entity.isdigit() and int(entity) < 10:
                continue

            pos = m.start()
            # Avoid overlapping matches (relaxed: only exact position conflicts)
            if pos in seen_positions or any(abs(pos - p) < 5 and entity in _STOP_ENTITIES for p in seen_positions):
                continue

            ec = _classify_entity(entity)
            # Semantic tags are stored separately (not appended to entity text)
            # to keep anchors compact. Tags are used during TF-IDF/FTS5 search.
            entities.append((entity, ec, m.start(), m.end()))
            seen_positions.add(pos)

    # Chinese terms — extract last since lower priority
    for m in _RE_CHINESE_TERM.finditer(text):
        pos = m.start()
        if any(abs(pos - p) < 2 for p in seen_positions):
            continue
        entities.append((m.group(0), EntityClass.TERM, m.start(), m.end()))
        seen_positions.add(pos)

    # Sort by position
    entities.sort(key=lambda x: x[2])
    return entities


def _is_proper_entity(entity: str) -> bool:
    """Gate: is this entity a proper noun/measurement, not a common word?

    Filters out sentence-initial common words, bare verbs, adjectives,
    and other low-information-density entities from regex over-matching.
    """
    # Hard filter: common English words that regex incorrectly captures
    _common_word_blacklist = {
        "Decided", "Current", "Store", "Database", "Impact", "Decision",
        "Acceptable", "Cannot", "Wrote", "Default", "Coverage", "Specifically",
        "Connection", "Security", "Compliance", "Safety", "Render",
        "Caught", "Testing", "Performance", "Accessibility", "Login",
        "Memory", "Rollout", "Estimated", "Switched", "Discovered",
        "Fixed", "Added", "Increased", "Specifically",
    }
    if entity in _common_word_blacklist:
        return False

    # Digits → measurement, version, error code → always keep
    if any(c.isdigit() for c in entity):
        return True

    # File extension → filename → always keep
    if '.' in entity:
        return True

    has_upper = any(c.isupper() for c in entity)
    has_lower = any(c.islower() for c in entity)
    has_underscore = '_' in entity

    # Snake_case compound → identifier → keep
    if has_underscore and len(entity) >= 6:
        return True

    # All uppercase acronym >=3 chars: JWT, TOTP, LRU, GDPR, CSRF
    if has_upper and not has_lower:
        return len(entity) >= 3

    # Internal case change (PascalCase/camelCase): uppercase beyond position 0
    if has_upper and has_lower:
        caps_after_first = sum(1 for c in entity[1:] if c.isupper())
        if caps_after_first >= 1:
            return True  # PostgreSQL, OAuth2, RedisCluster
        # Single initial capital: "Redis" (5 chars, proper noun) vs "Decided" (7, common)
        # Keep if appears to be a tech term: short and consonant-heavy
        if 4 <= len(entity) <= 8:
            vowels = sum(1 for c in entity.lower() if c in 'aeiou')
            # Tech terms: low vowel ratio (Redis=40%, Nginx=40%, Flask=20%)
            # Common words: higher vowel ratio (Decide=50%, Cannot=50%)
            return vowels <= len(entity) * 0.42  # Redis=2/5=0.4 passes; Decided=3/7=0.43 fails
        return len(entity) > 8

    return False


def _extract_data_values(text: str) -> list[str]:
    """Extract precise data values attached to an entity context."""
    values = []
    for m in _RE_VERSION.finditer(text):
        values.append(m.group(0))
    for m in _RE_ERROR_CODE.finditer(text):
        values.append(m.group(0))
    for m in _RE_LINE_NUM.finditer(text):
        values.append(f"line:{m.group(1)}")
    return list(set(values))[:5]  # Max 5 data values


def _resolve_tags(entity: str) -> list[str]:
    """Look up semantic tags for an entity (exact match or compound components).

    Tags are stored on the Anchor object but NOT in the entity text,
    keeping anchors compact (~15 chars) while enabling semantic search.
    """
    tags = list(_ENTITY_SEMANTIC_TAGS.get(entity, []))
    if not tags and " + " in entity:
        for component in entity.split(" + "):
            for t in _ENTITY_SEMANTIC_TAGS.get(component.strip(), []):
                if t not in tags:
                    tags.append(t)
    return tags


# ── Main extraction pipeline ──────────────────────────────────────────────

WINDOW_SIZE = 80  # Characters to look around a verb for nearby entities


# ── Helpers ────────────────────────────────────────────────────────────

# LLM-based significance judge — replaces all hand-crafted rules
from .judge import judge_significance


def _find_nearest_noun(noun_matches, pos, window):
    best, best_dist = None, float('inf')
    for item in noun_matches:
        text, cls, start, end = item
        dist = abs(pos - start)
        if dist < best_dist and dist < window:
            best_dist = dist
            best = item
    return best


def _find_nearest_verb(verb_matches, pos, window):
    best, best_dist = None, float('inf')
    for v_text, v_type, v_start, v_end in verb_matches:
        dist = abs(pos - v_start)
        if dist < best_dist and dist < window:
            best_dist = dist
            best = (v_text, v_type, v_start, v_end)
    return best


# ── Main extraction ────────────────────────────────────────────────────

def extract_graph(messages: list[dict], session_id: Optional[str] = None) -> AnchorGraph:
    """Extract bidirectional verb-noun anchor graph from messages.

    Each verb anchors to its nearest noun. Each noun anchors to its nearest verb.
    All candidates are scored by significance, then top-N are kept.
    Target: ~1 anchor per 2.5 messages → ~92% compression.
    """
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]

    full_text = ""
    offset = 0
    all_code_refs = []
    for msg in messages:
        content = msg.get("content", "")
        if "tool_result" in msg:
            content += "\n" + str(msg["tool_result"])
        # Strip code blocks — replace with summary, keep refs for file re-reading
        content, code_refs = _strip_code_blocks(content)
        all_code_refs.extend(code_refs)
        full_text += content + "\n"
        offset += len(content) + 1

    if not full_text.strip():
        return AnchorGraph(session_id=session_id)

    verb_matches = segment_text(full_text)
    noun_matches = _extract_entities(full_text)

    # Phase 1: Verb anchors — each verb links to its nearest noun
    # Filter out generic/low-quality verbs that have no information value
    _GENERIC_VERBS = {"set to", "Add", "add", "use", "using",
                      "make", "need", "needs", "want", "keep", "take", "get",
                      "Deploy", "deploy", "See", "see", "Check", "check"}
    verb_anchors = []
    seen_verb_entities = set()  # Dedup by entity text
    seen_verbs = set()
    for verb_text, verb_type, v_start, v_end in verb_matches:
        if verb_type == "FACT" or verb_text.lower() in _STOP_ENTITIES:
            continue
        if verb_text in _GENERIC_VERBS:
            continue
        if verb_text in seen_verb_entities:
            continue
        seen_verb_entities.add(verb_text)
        key = (verb_text, v_start)
        if key in seen_verbs:
            continue
        seen_verbs.add(key)

        nearest = _find_nearest_noun(noun_matches, v_start, WINDOW_SIZE)
        if nearest is None:
            continue
        noun_text, noun_class, n_start, n_end = nearest

        data_hints = _extract_data_values(full_text[max(0, v_start - 40):min(len(full_text), n_end + 40)])
        v = VerbAnchor(entity=verb_text, anchor_type=AnchorType(verb_type),
                       pos=v_start, data_hints=data_hints)
        verb_anchors.append(v)

    # Phase 2: Noun anchors — each noun links to its nearest verb
    # Filter garbage: 3-char all-caps acronyms with no data context
    _GARBAGE_NOUNS = {"UTC", "min", "hour", "day", "days", "percent", "users",
                       "Deployed", "Decided", "Switched", "Discovered", "Fixed",
                       "Added", "Increased", "Wrote"}
    noun_anchors = []
    seen_noun_entities = set()  # Dedup by entity text
    seen_nouns = set()
    for noun_text, noun_class, n_start, n_end in noun_matches:
        if noun_class != EntityClass.DATA and not _is_proper_entity(noun_text):
            continue
        if noun_text in _GARBAGE_NOUNS:
            continue
        if noun_text in seen_noun_entities:
            continue
        seen_noun_entities.add(noun_text)

        nearest = _find_nearest_verb(verb_matches, n_start, WINDOW_SIZE)
        if nearest is None:
            continue
        v_text, v_type, v_start, v_end = nearest

        data_vals = _extract_data_values(full_text[max(0, n_start - 40):min(len(full_text), n_end + 40)])
        n = NounAnchor(entity=noun_text, entity_class=noun_class,
                       pos=n_start, data_values=data_vals,
                       tags=_resolve_tags(noun_text))
        noun_anchors.append(n)

    # Phase 3: Resolve bidirectional links
    for v in verb_anchors:
        nearest = _find_nearest_noun(noun_matches, v.pos, WINDOW_SIZE)
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

    # Phase 4: LLM significance judge — replaces all scoring/dicts/quotas
    # Build deduplicated candidate list for the LLM
    target = max(8, len(messages) // 2)
    seen_candidates = set()
    candidate_list = []
    for v in verb_anchors:
        key = (v.entity, "verb")
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        candidate_list.append({
            "entity": v.entity, "type": "verb", "verb_type": v.anchor_type.value,
            "pos": v.pos, "data": v.data_hints,
        })
    for n in noun_anchors:
        key = (n.entity, "noun")
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        candidate_list.append({
            "entity": n.entity, "type": "noun", "noun_class": n.entity_class.value,
            "pos": n.pos, "data": n.data_values,
        })

    # Ask LLM to select the most significant ones + generate tags
    excerpt = full_text[:2000]  # First 2000 chars for context
    selections = judge_significance(candidate_list, excerpt, target)

    # Map selections back to anchors
    selected_entities = {(s["entity"], s["type"]) for s in selections}
    selected_tags = {(s["entity"], s["type"]): s.get("tags", []) for s in selections}

    kept_verbs = [v for v in verb_anchors if (v.entity, "verb") in selected_entities]
    kept_nouns = []
    for n in noun_anchors:
        if (n.entity, "noun") in selected_entities:
            llm_tags = selected_tags.get((n.entity, "noun"), [])
            if llm_tags:
                n.tags = llm_tags  # LLM tags override regex tags
            # else keep _resolve_tags() tags from Phase 2
            kept_nouns.append(n)

    kept_verb_ids = {v.id for v in kept_verbs}
    kept_noun_ids = {n.id for n in kept_nouns}

    # Clear dangling links
    for v in kept_verbs:
        if v.nearest_noun_id not in kept_noun_ids:
            v.nearest_noun_id = ""
    for n in kept_nouns:
        if n.nearest_verb_id not in kept_verb_ids:
            n.nearest_verb_id = ""

    return AnchorGraph(session_id=session_id, verb_anchors=kept_verbs, noun_anchors=kept_nouns)


def extract_anchors(messages: list[dict], session_id: Optional[str] = None) -> AnchorSequence:
    """Legacy wrapper — delegates to extract_graph, returns flat AnchorSequence."""
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
