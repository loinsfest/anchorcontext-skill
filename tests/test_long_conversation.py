# -*- coding: utf-8 -*-
"""Long conversation E2E test — extract -> reconstruct -> score.

This simulates a realistic 30-message development session covering:
- Architecture decisions (Redis, JWT, PostgreSQL)
- Bug discoveries (race condition, mutex, timeout)
- Constraints (cross-pod sync, rate limiting)
- Facts (version numbers, error codes, pool sizes)
"""

import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))

from anchor.extractor import extract_anchors
from anchor.store import AnchorStore
from anchor.store_sqlite import SqliteStore
from anchor.reconstructor import SequenceRetriever, HybridRetriever
from anchor.formatter import format_for_injection, format_compact

# ============================================================
# TEST DATA: 30-message realistic development conversation
# ============================================================
# Each message has an id for ground-truth referencing
CONVERSATION = [
    # --- ARCHITECTURE PHASE ---
    {"id": 1, "content": "We need to build a user authentication system for the new microservice. Current system is monolithic and can't scale."},
    {"id": 2, "content": "I propose using JWT tokens for stateless auth. Store the access token in memory, refresh token in an httpOnly cookie."},
    {"id": 3, "content": "Wait — localStorage has XSS risk. Let's use httpOnly cookies for both tokens. CSRF protection via SameSite=Strict."},
    {"id": 4, "content": "Decided to use Redis for session management. Specifically Redis SETNX for distributed locking across multiple auth service instances."},
    {"id": 5, "content": "Database will be PostgreSQL 14.2. Connection pool size set to 20 with PgBouncer in front for transaction pooling."},

    # --- BUG DISCOVERY PHASE ---
    {"id": 6, "content": "Found a critical race condition in auth.ts at line 42. JWT refresh logic — two simultaneous refresh requests can both succeed, creating duplicate tokens."},
    {"id": 7, "content": "Error code for this is ERR_005. Root cause: the refresh endpoint checks token validity BEFORE acquiring the Redis lock, not after."},
    {"id": 8, "content": "Also discovered that the mutex-based locking in the old auth system doesn't work across Pods. Only protects within a single process."},
    {"id": 9, "content": "This explains the intermittent 401 errors we saw in production — pods were invalidating each other's tokens."},

    # --- FIX IMPLEMENTATION ---
    {"id": 10, "content": "Fix: acquire Redis SETNX lock FIRST, then check token validity. Lock timeout set to 5 seconds. Key format: 'refresh_lock:{userId}'."},
    {"id": 11, "content": "Added rate limiting per IP: max 100 requests per minute. Using sliding window algorithm in Redis."},
    {"id": 12, "content": "Deployed the fix to staging. API response time dropped from 200ms to 80ms after the lock ordering fix."},
    {"id": 13, "content": "But we discovered a new issue: Redis connection pool exhaustion at 50 concurrent users. Default pool size of 10 is too small."},
    {"id": 14, "content": "Increased Redis connection pool to 50, added connection timeout of 3 seconds, and enabled TCP keepalive."},

    # --- MONITORING & OBSERVABILITY ---
    {"id": 15, "content": "Added Prometheus metrics for auth service: auth_requests_total, auth_latency_seconds, redis_pool_active_connections."},
    {"id": 16, "content": "Grafana dashboard at grafana.internal/d/auth-latency is what oncall watches. Set alert threshold at p95 > 200ms."},
    {"id": 17, "content": "Discovered that the /api/login endpoint is 3x slower than /api/refresh. Login does a bcrypt compare (cost factor 12) which takes ~250ms."},
    {"id": 18, "content": "Decision: reduced bcrypt cost factor from 12 to 10. Security team approved — still within NIST guidelines for our threat model."},

    # --- CROSS-CUTTING CONCERNS ---
    {"id": 19, "content": "Must ensure GDPR compliance for EU users. Auth tokens must be deletable within 30 days per right-to-erasure requests."},
    {"id": 20, "content": "Added a token blacklist in Redis with TTL matching the token expiry. Blacklist checked on every authenticated request."},
    {"id": 21, "content": "Performance impact of blacklist check: 2ms overhead per request. Acceptable per our SLA of p95 < 100ms for auth endpoints."},
    {"id": 22, "content": "Cannot use JWT alone for sensitive operations — must add a second factor (TOTP) for admin panel access. Compliance requirement."},

    # --- TESTING & DEPLOYMENT ---
    {"id": 23, "content": "Wrote 47 integration tests for the auth service. Coverage: 92% lines, 88% branches. Main gap is OAuth2 social login flow."},
    {"id": 24, "content": "Load test results: 500 RPS sustained for 10 minutes. p50=45ms, p95=88ms, p99=150ms. Zero errors. Redis CPU at 15%."},
    {"id": 25, "content": "Discovered memory leak in the token cache layer. LRU eviction was broken — max size 10000 but it grew to 85000 entries."},
    {"id": 26, "content": "Root cause: the LRU counter was using a 32-bit int that overflowed at ~2 billion operations. Switched to 64-bit. Memory usage dropped from 2.1GB to 180MB."},
    {"id": 27, "content": "Deployed to production at 15:21 UTC. Rollout: 10% canary for 30 min, then 50% for 1 hour, then 100%. No alerts fired."},

    # --- POST-MORTEM & FUTURE ---
    {"id": 28, "content": "Post-mortem findings: the race condition bug (ERR_005) was present for 14 days before detection. Impact: ~3% of users experienced intermittent logout."},
    {"id": 29, "content": "Decision: add integration tests that specifically test concurrent token refresh scenarios. Use Redis TEST harness, not mocks."},
    {"id": 30, "content": "Next sprint: OAuth2 Google/GitHub social login. Estimated 8 story points. Must maintain the same Redis session architecture."},
]

# ============================================================
# GROUND TRUTH: Expected answers for specific queries
# ============================================================
GROUND_TRUTH = {
    "What is the Redis locking approach?": {
        "must_contain": ["Redis", "SETNX", "distributed lock", "5 seconds"],
        "should_contain": ["refresh_lock", "userId"],
        "key_message_ids": [4, 10],
        "score_weight": 1.0,
    },
    "What bug was found in auth.ts?": {
        "must_contain": ["auth.ts", "line 42", "race condition", "JWT", "ERR_005"],
        "should_contain": ["refresh", "simultaneous", "duplicate tokens"],
        "key_message_ids": [6, 7],
        "score_weight": 1.0,
    },
    "What database and versions are used?": {
        "must_contain": ["PostgreSQL", "14.2"],
        "should_contain": ["pool size", "20", "PgBouncer"],
        "key_message_ids": [5],
        "score_weight": 0.8,
    },
    "What performance issues were discovered?": {
        "must_contain": ["200ms", "80ms", "memory leak", "LRU"],
        "should_contain": ["bcrypt", "cost factor", "2.1GB", "180MB"],
        "key_message_ids": [12, 17, 25, 26],
        "score_weight": 1.2,
    },
    "What happened with the memory leak?": {
        "must_contain": ["LRU", "32-bit", "overflow", "2.1GB", "180MB"],
        "should_contain": ["token cache", "10000", "85000"],
        "key_message_ids": [25, 26],
        "score_weight": 1.0,
    },
}


def score_reconstruction(anchor_text: str, query: str, gt: dict) -> dict:
    """Score reconstruction against ground truth."""
    must_hits = sum(1 for term in gt["must_contain"] if term.lower() in anchor_text.lower())
    should_hits = sum(1 for term in gt["should_contain"] if term.lower() in anchor_text.lower())

    must_score = must_hits / len(gt["must_contain"]) * 10 if gt["must_contain"] else 10
    should_bonus = (should_hits / len(gt["should_contain"])) * 2 if gt["should_contain"] else 0

    total = min(10, must_score + should_bonus) * gt["score_weight"]

    return {
        "query": query,
        "must_hits": f"{must_hits}/{len(gt['must_contain'])}",
        "should_hits": f"{should_hits}/{len(gt['should_contain'])}",
        "raw_score": round(min(10, must_score + should_bonus), 1),
        "weighted_score": round(total, 1),
        "weight": gt["score_weight"],
    }


def main():
    print("=" * 70)
    print("  Anchor Context — Long Conversation E2E Test")
    print("=" * 70)
    print(f"  Input: {len(CONVERSATION)} messages")
    total_chars = sum(len(m["content"]) for m in CONVERSATION)
    print(f"  Total text: {total_chars} characters (~{total_chars // 4} tokens)")
    print()

    # ====== STEP 1: Extract ======
    print("[STEP 1] Extracting anchors...")
    seq = extract_anchors(CONVERSATION)
    active = seq.get_active()
    compression_ratio = (1 - len(active) * 15 / total_chars) * 100  # ~15 chars per anchor
    print(f"  Anchors extracted: {len(seq.anchors)} total, {len(active)} active")
    print(f"  Approx compression: {compression_ratio:.1f}%")
    print()

    # Show extracted anchors by type
    from collections import Counter
    type_counts = Counter(a.anchor_type.value for a in active)
    for atype, count in type_counts.most_common():
        print(f"    {atype}: {count}")

    print()
    print("  Top anchors by type:")
    for atype in ["DECISION", "DISCOVERY", "ANOMALY", "CONSTRAINT"]:
        typed = [a for a in active if a.anchor_type.value == atype][:4]
        for a in typed:
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            print(f"    [{a.anchor_type.value:12s}] {a.entity[:50]}{dv}")

    # ====== STEP 2: Save to SQLite ======
    print()
    print("[STEP 2] Saving to SQLite (FTS5)...")
    sqlite = SqliteStore()
    sqlite.save_sequence(seq)
    loaded = sqlite.load_sequence(seq.session_id)
    print(f"  SQLite reload: {len(loaded.get_active())} active anchors")
    fts_test = sqlite.search("Redis", limit=3)
    print(f"  FTS5 search 'Redis': {len(fts_test)} results")
    for r in fts_test:
        print(f"    [{r['anchor_type']}] {r['entity'][:60]}")

    # ====== STEP 3: Reconstruction queries ======
    print()
    print("[STEP 3] Running reconstruction queries...")
    print("-" * 70)

    tfidf = SequenceRetriever(seq)
    hybrid = HybridRetriever(seq)
    results = []

    for query, gt in GROUND_TRUTH.items():
        # TF-IDF retrieval
        _, hit_idx, tfidf_score = tfidf.find_position(query)
        window = tfidf.get_window(hit_idx, radius=2)

        # Build anchor text from window
        anchor_text = " ".join(a.entity for a in window)

        # Add data values
        for a in window:
            if a.data_values:
                anchor_text += " " + " ".join(a.data_values)

        # FTS5 fallback
        fts_results = sqlite.search(query, limit=3) if tfidf_score < 0.15 else None

        # Score against ground truth
        result = score_reconstruction(anchor_text, query, gt)

        # Hybrid prompt
        prompt = hybrid.search(query, radius=2)

        result["tfidf_score"] = round(tfidf_score, 3)
        result["window_size"] = len(window)
        result["fts_fallback"] = len(fts_results) if fts_results else 0
        result["prompt_chars"] = len(prompt)
        results.append(result)

        print(f"\n  Query: '{query}'")
        print(f"    TF-IDF score: {tfidf_score:.3f} | Window: {len(window)} anchors")
        print(f"    Must: {result['must_hits']} | Should: {result['should_hits']}")
        print(f"    Raw: {result['raw_score']}/10 | Weighted: {result['weighted_score']:.1f}")
        print(f"    FTS5 fallback results: {result['fts_fallback']}")

        # Show the matched anchors
        print(f"    Window anchors:")
        for i, a in enumerate(window):
            marker = " [PRIMARY]" if a.pos == active[hit_idx].pos else ""
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            print(f"      {a.entity[:60]}{dv}{marker}")

    # ====== STEP 4: Overall scores ======
    print()
    print("=" * 70)
    print("  OVERALL SCORES")
    print("=" * 70)
    print()
    print(f"  {'Query':<45s} {'Raw':>5s} {'Weighted':>9s} {'TF-IDF':>7s} {'FTS5':>5s}")
    print(f"  {'-'*45} {'-'*5} {'-'*9} {'-'*7} {'-'*5}")

    total_weighted = 0
    total_weight = 0
    for r in results:
        print(f"  {r['query'][:44]:<45s} {r['raw_score']:>4.1f}  {r['weighted_score']:>7.1f}  {r['tfidf_score']:>5.3f}  {r['fts_fallback']:>4d}")
        total_weighted += r['weighted_score']
        total_weight += r['weight']

    avg_weighted = total_weighted / total_weight if total_weight > 0 else 0
    print(f"  {'-'*45} {'-'*5} {'-'*9} {'-'*7} {'-'*5}")
    print(f"  {'WEIGHTED AVERAGE':<45s}        {avg_weighted:>7.1f}")
    print()

    # ====== STEP 5: Analysis ======
    print("=" * 70)
    print("  ANALYSIS & IMPROVEMENT AREAS")
    print("=" * 70)
    print()

    # What worked well
    print("  [STRENGTHS]")
    low_score_queries = [r for r in results if r['raw_score'] < 7]
    high_score_queries = [r for r in results if r['raw_score'] >= 7]

    print(f"    High-scoring queries (>=7): {len(high_score_queries)}/{len(results)}")
    for r in high_score_queries:
        print(f"      + {r['query'][:50]}: {r['raw_score']}/10 — TF-IDF={r['tfidf_score']:.3f}")

    print(f"    Low-scoring queries (<7): {len(low_score_queries)}/{len(results)}")
    for r in low_score_queries:
        print(f"      - {r['query'][:50]}: {r['raw_score']}/10 — TF-IDF={r['tfidf_score']:.3f}, FTS5={r['fts_fallback']}")

    # What extracted well
    print()
    print("  [EXTRACTION QUALITY]")
    data_anchors = [a for a in active if a.entity_class.value == "DATA"]
    tech_anchors = [a for a in active if a.entity_class.value == "TECH"]
    term_anchors = [a for a in active if a.entity_class.value == "TERM"]
    print(f"    DATA anchors (numbers/versions/codes): {len(data_anchors)}")
    print(f"    TECH anchors (filenames/identifiers):  {len(tech_anchors)}")
    print(f"    TERM anchors (concepts):               {len(term_anchors)}")

    # Check specific expected entities
    expected_data = ["14.2", "42", "ERR_005", "20", "50", "200", "80", "2.1GB", "180MB", "100", "10", "12"]
    found_data = []
    for ed in expected_data:
        for a in active:
            if ed.lower() in a.entity.lower() or (a.data_values and ed.lower() in str(a.data_values).lower()):
                found_data.append(ed)
                break
    print(f"    Critical data values captured: {len(found_data)}/{len(expected_data)}")
    print(f"    Captured: {found_data}")
    missing = [e for e in expected_data if e not in found_data]
    if missing:
        print(f"    MISSING: {missing}")

    # Expected tech entities
    expected_tech = ["Redis", "SETNX", "JWT", "auth.ts", "PostgreSQL", "PgBouncer", "Prometheus", "Grafana", "LRU", "TOTP", "OAuth2"]
    found_tech = []
    for et in expected_tech:
        for a in active:
            if et.lower() in a.entity.lower():
                found_tech.append(et)
                break
    print(f"    Critical tech entities captured: {len(found_tech)}/{len(expected_tech)}")
    missing_tech = [e for e in expected_tech if e not in found_tech]
    if missing_tech:
        print(f"    MISSING: {missing_tech}")

    print()
    print("  [IMPROVEMENT AREAS]")
    issues = []

    if missing:
        issues.append(f"DATA values not extracted: {missing} — pure numbers like 20, 50, 100 need unit context")
    if missing_tech:
        issues.append(f"TECH entities missed: {missing_tech} — some identifiers lost during extraction")
    if low_score_queries:
        for r in low_score_queries:
            if r['tfidf_score'] < 0.1:
                issues.append(f"TF-IDF near-zero for '{r['query'][:40]}' — Chinese/semantic gap not bridged by FTS5")

    for issue in issues:
        print(f"    1. {issue}")

    # Compression stats
    anchor_chars = sum(len(a.entity) + 15 for a in active)  # entity + overhead
    print()
    print(f"  [COMPRESSION STATS]")
    print(f"    Original: {total_chars} chars (~{total_chars // 4} tokens)")
    print(f"    Anchors:  {anchor_chars} chars (~{anchor_chars // 4} tokens)")
    print(f"    Ratio:    {(1 - anchor_chars/total_chars) * 100:.1f}% reduction")

    print()
    print("=" * 70)
    print(f"  FINAL VERDICT: {avg_weighted:.1f}/10 weighted average")
    if avg_weighted >= 8:
        print("  Production ready — excellent reconstruction fidelity")
    elif avg_weighted >= 6:
        print("  Good — usable with room for improvement")
    else:
        print("  Needs work — reconstruction fidelity below threshold")
    print("=" * 70)


if __name__ == "__main__":
    main()
