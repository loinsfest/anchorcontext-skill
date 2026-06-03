# -*- coding: utf-8 -*-
"""Second long conversation E2E test — frontend architecture + system design.

Tests generalization beyond backend/auth domain. Validates semantic tags
and extraction quality across different entity types.
"""

import sys, os, json, math
from collections import Counter
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))

from anchor.extractor import extract_anchors
from anchor.store import AnchorStore
from anchor.store_sqlite import SqliteStore
from anchor.reconstructor import SequenceRetriever, HybridRetriever
from anchor.formatter import format_for_injection

# ============================================================
# 40-message frontend + system design conversation
# ============================================================
CONVERSATION = [
    # --- FRONTEND ARCHITECTURE ---
    {"id": 1, "content": "We need to migrate our dashboard from React class components to functional components with hooks. Currently on React 16.8, targeting React 18.3."},
    {"id": 2, "content": "State management: deciding between Zustand and Redux Toolkit. Zustand is 1.1KB gzipped vs Redux at 11KB. For a dashboard with 50 API endpoints, Zustand is sufficient."},
    {"id": 3, "content": "Decided to use Zustand 4.5 with immer middleware for immutable state updates. Added devtools middleware for debugging in Chrome DevTools."},
    {"id": 4, "content": "CSS approach: using Tailwind CSS 3.4 with the @apply directive for component classes. Avoid inline style objects — they break React's reconciliation for complex layouts."},
    {"id": 5, "content": "Component library: building custom components with Radix UI primitives. Headless UI approach — unstyled accessible components that we style with Tailwind."},

    # --- PERFORMANCE OPTIMIZATION ---
    {"id": 6, "content": "Lighthouse audit shows Core Web Vitals: LCP at 3.2s, CLS at 0.15, INP at 280ms. All fail Google's 'good' thresholds. Main culprit: unoptimized API waterfall."},
    {"id": 7, "content": "Fixed LCP by implementing React.lazy + Suspense for route-level code splitting. Bundle size dropped from 2.4MB to 890KB (gzipped: 320KB). LCP improved to 1.8s."},
    {"id": 8, "content": "Discovered a memory leak in the WebSocket reconnection logic. The useEffect cleanup wasn't calling ws.close() on unmount, leaving zombie connections accumulating at ~50MB per hour."},
    {"id": 9, "content": "Fixed WebSocket leak by adding proper cleanup in useEffect return. Memory usage stabilized at 120MB baseline (was growing to 800MB after 8 hours)."},
    {"id": 10, "content": "Performance regression found: the Zustand selector `useStore(s => s.items)` was causing full re-renders. Changed to shallow comparison with `useStore(s => s.items, shallow)`. Render count dropped 70%."},

    # --- TESTING INFRASTRUCTURE ---
    {"id": 11, "content": "Testing stack: Vitest 1.6 for unit tests, Playwright 1.45 for E2E. Vitest is 4x faster than Jest because it reuses Vite's transform pipeline."},
    {"id": 12, "content": "Unit test coverage: 88% lines, 82% branches. Main gap is error boundary components and WebSocket reconnection scenarios. Need to add Mock Service Worker (MSW) for API mocking."},
    {"id": 13, "content": "Playwright E2E: 34 tests covering critical paths (login flow, dashboard rendering, data export). CI runs in GitHub Actions on Chromium, Firefox, and WebKit. Average E2E suite: 4m 12s."},
    {"id": 14, "content": "Found a flaky test in the data table pagination spec. The test was relying on setTimeout(500) for loading state. Replaced with `page.waitForSelector('[data-testid=\"row-1\"]')`. Flakiness dropped from 15% to 0%."},

    # --- CI/CD PIPELINE ---
    {"id": 15, "content": "CI/CD: GitHub Actions with 3 workflows — lint+test on PR, visual regression on Chromatic, deploy preview on Vercel. Average PR pipeline: 6m 30s."},
    {"id": 16, "content": "Build optimization: switched from Webpack to Vite 5.4. Dev server cold start: 45s → 2.3s. HMR: 800ms → 20ms. Build: 4m → 38s."},
    {"id": 17, "content": "Docker image size issue: Node alpine image was 680MB with all dev dependencies. Multi-stage build reduced it to 145MB. Used pnpm instead of npm for another 30% reduction in layer caching."},
    {"id": 18, "content": "Decision: adopt pnpm workspace monorepo structure. Shared packages: @acme/ui (components), @acme/utils (helpers), @acme/config (eslint, tsconfig, prettier). Zero-hoist for strict dependency isolation."},

    # --- API DESIGN ---
    {"id": 19, "content": "API layer: using tRPC 11 with React Query (TanStack Query v5) for type-safe API calls. Zero manual type definitions — types flow from Prisma schema → tRPC router → React hooks."},
    {"id": 20, "content": "Database: decided on PlanetScale (MySQL 8.0 compatible) with Prisma ORM 5.14. Connection pooling handled by Prisma Data Proxy — p95 query latency at 12ms from us-east-1."},
    {"id": 21, "content": "Rate limiting: implemented token bucket algorithm in Cloudflare Workers. 1000 req/min per user, 5000 req/min per IP. Override for enterprise tier: 10000 req/min."},
    {"id": 22, "content": "Discovered N+1 query problem in the dashboard aggregations endpoint. Prisma's `include` was doing separate queries for each dashboard item. Fixed with `findMany` + manual joins using Prisma's `$queryRaw`. Query count dropped from 251 to 4."},

    # --- SECURITY ---
    {"id": 23, "content": "Authentication: Clerk (formerly Clerk.dev) for React. Session tokens stored in httpOnly cookies. JWT verification happens at the edge via Cloudflare Workers middleware."},
    {"id": 24, "content": "CSRF protection: double-submit cookie pattern. Server generates a random token, client sends it back in both cookie and custom header. Comparison happens at edge — zero latency overhead."},
    {"id": 25, "content": "Security audit found: the file upload endpoint accepted arbitrary MIME types. Added server-side validation with Magic Number checking (read first 4 bytes). Blocked SVG+XSS, polyglot PNG uploads."},
    {"id": 26, "content": "CSP headers: added Content-Security-Policy with strict-dynamic for scripts, 'self' for images, and report-uri endpoint for violation reporting. Zero inline scripts — all scripts loaded from trusted CDN."},

    # --- OBSERVABILITY ---
    {"id": 27, "content": "Monitoring: Datadog RUM for frontend, Datadog APM for backend. Custom dashboard tracking Core Web Vitals P95, API latency P99, and error rates by route."},
    {"id": 28, "content": "Alerting: PagerDuty integration. Critical alerts: error rate > 1% for 5 min, P95 API latency > 500ms for 3 min, 5xx status code count > 10 in 1 min. On-call rotation: 3 engineers, weekly."},
    {"id": 29, "content": "Logging: structured JSON logging with Winston 3.11 → shipped to Datadog via agent. Log levels: error (critical), warn (degradation), info (business events), debug (development only). PII redaction in production via custom formatter."},
    {"id": 30, "content": "Discovered that the error tracking was missing unhandled Promise rejections in async event handlers. Added `window.addEventListener('unhandledrejection')` with Sentry capture. Caught 23 production errors in first week that were previously silent."},

    # --- ACCESSIBILITY ---
    {"id": 31, "content": "Accessibility audit with axe-core 4.9: 47 violations. Main issues: missing aria-labels on icon buttons, insufficient color contrast (ratio < 4.5:1), keyboard trap in modal dialogs."},
    {"id": 32, "content": "Fixed keyboard navigation: Tab order follows visual layout. Focus trap in modals using focus-trap-react 10.0. Skip-to-content link added as first focusable element. Screen reader testing with VoiceOver + NVDA."},
    {"id": 33, "content": "Color contrast: increased from 3.2:1 to 4.8:1 for body text. Dark mode palette adjusted: gray-600 → gray-500 for better legibility. All interactive elements now have visible focus rings (2px offset, 3px blur, blue-500)."},

    # --- MOBILE & RESPONSIVE ---
    {"id": 34, "content": "Responsive design: mobile-first breakpoints at 640px (sm), 768px (md), 1024px (lg), 1280px (xl). Dashboard switches from side navigation to bottom tab bar at < 768px."},
    {"id": 35, "content": "Mobile performance: separate code bundle for mobile via Vite's conditional builds. Mobile bundle is 40% smaller by excluding desktop-only features like drag-and-drop and multi-column layouts."},
    {"id": 36, "content": "Touch interactions: replaced hover-dependent dropdowns with press-and-hold patterns on mobile. Added swipe gestures (react-swipeable 7.0) for table row actions. Minimum touch target: 44x44px per WCAG 2.2 AAA."},

    # --- FUTURE ROADMAP ---
    {"id": 37, "content": "Next quarter: migrate from REST to GraphQL using Apollo Client 4 + GraphQL Code Generator. Estimated 12 story points, 3 sprints. GraphQL schema already designed — 38 types, 52 queries, 18 mutations."},
    {"id": 38, "content": "A/B testing infrastructure: evaluating LaunchDarkly vs self-hosted Flagsmith. Flagsmith chosen for GDPR compliance (EU data residency). Feature flags with gradual rollout: 5% → 25% → 50% → 100%."},
    {"id": 39, "content": "Design system: building a shared component library with Storybook 8.0. Currently 47 components documented. Goal: 80 components by Q3 with automated visual regression via Chromatic. Zero visual regressions policy."},
    {"id": 40, "content": "Team growth: hiring 2 senior frontend engineers (React + TypeScript), 1 platform engineer (CI/CD + infra). Tech stack documentation in Notion. Onboarding time target: 2 weeks to first PR."},
]

# ============================================================
# GROUND TRUTH
# ============================================================
GROUND_TRUTH = {
    "What state management library was chosen and why?": {
        "must_contain": ["Zustand", "Redux", "1.1KB", "immer"],
        "should_contain": ["devtools", "shallow"],
        "key_message_ids": [2, 3, 10],
        "score_weight": 1.0,
    },
    "How was the WebSocket memory leak fixed?": {
        "must_contain": ["WebSocket", "cleanup", "useEffect", "120MB", "800MB"],
        "should_contain": ["ws.close", "unmount", "zombie"],
        "key_message_ids": [8, 9],
        "score_weight": 1.0,
    },
    "What testing tools and infrastructure are used?": {
        "must_contain": ["Vitest", "Playwright", "MSW"],
        "should_contain": ["coverage", "88%", "flaky"],
        "key_message_ids": [11, 12, 13, 14],
        "score_weight": 1.2,
    },
    "What build and CI/CD setup is used?": {
        "must_contain": ["Vite", "GitHub Actions", "Vercel"],
        "should_contain": ["Docker", "145MB", "pnpm"],
        "key_message_ids": [15, 16, 17, 18],
        "score_weight": 1.0,
    },
    "What database and API architecture is used?": {
        "must_contain": ["PlanetScale", "Prisma", "tRPC", "TanStack"],
        "should_contain": ["N+1", "251", "12ms"],
        "key_message_ids": [19, 20, 22],
        "score_weight": 1.0,
    },
    "What accessibility improvements were made?": {
        "must_contain": ["aria-labels", "color contrast", "focus trap", "WCAG"],
        "should_contain": ["4.8:1", "44x44", "skip-to-content"],
        "key_message_ids": [31, 32, 33, 36],
        "score_weight": 1.0,
    },
}


def score_reconstruction(anchor_text, query, gt):
    must_hits = sum(1 for t in gt["must_contain"] if t.lower() in anchor_text.lower())
    should_hits = sum(1 for t in gt["should_contain"] if t.lower() in anchor_text.lower())
    raw = min(10, (must_hits / len(gt["must_contain"])) * 10 + (should_hits / max(1, len(gt["should_contain"]))) * 2)
    weighted = raw * gt["score_weight"]
    return {"must": f"{must_hits}/{len(gt['must_contain'])}", "should": f"{should_hits}/{len(gt['should_contain'])}",
            "raw": round(min(10, raw), 1), "weighted": round(weighted, 1), "weight": gt["score_weight"]}


def main():
    print("=" * 70)
    print("  Anchor Context — E2E Test v2 (Frontend + System Design)")
    print("=" * 70)
    total_chars = sum(len(m["content"]) for m in CONVERSATION)
    print(f"  Input: {len(CONVERSATION)} messages, {total_chars} chars (~{total_chars // 4} tokens)")

    # ====== EXTRACT ======
    seq = extract_anchors(CONVERSATION)
    active = seq.get_active()
    type_counts = Counter(a.anchor_type.value for a in active)
    print(f"  Anchors: {len(active)} active ({', '.join(f'{t}:{c}' for t,c in type_counts.most_common(4))})")
    print()

    # Preview top anchors
    print("  Sample anchors:")
    for atype in ["DECISION", "DISCOVERY", "ANOMALY", "CONSTRAINT"]:
        typed = [a for a in active if a.anchor_type.value == atype and len(a.entity) > 10][:3]
        for a in typed:
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            print(f"    [{a.anchor_type.value:12s}] {a.entity[:70]}{dv}")

    # ====== SAVE ======
    sqlite = SqliteStore()
    sqlite.save_sequence(seq)

    # ====== RECONSTRUCT ======
    tfidf = SequenceRetriever(seq)
    hybrid = HybridRetriever(seq)
    results = []

    print(f"\n{'='*70}")
    print("  RECONSTRUCTION RESULTS")
    print(f"{'='*70}")

    for query, gt in GROUND_TRUTH.items():
        _, hit_idx, tfidf_score = tfidf.find_position(query)
        window = tfidf.get_window(hit_idx, radius=2)
        anchor_text = " ".join(a.entity for a in window)
        for a in window:
            if a.data_values:
                anchor_text += " " + " ".join(a.data_values)

        result = score_reconstruction(anchor_text, query, gt)
        fts_results = sqlite.search(" OR ".join(query.split()[:5]), limit=3) if tfidf_score < 0.15 else None

        result["tfidf"] = round(tfidf_score, 3)
        result["window"] = len(window)
        result["fts"] = len(fts_results) if fts_results else 0
        results.append(result)

        marker = " *FTS5 fallback*" if result["fts"] > 0 else ""
        print(f"\n  Q: {query}")
        print(f"     Score: {result['raw']}/10 | TF-IDF: {result['tfidf']} | Win: {result['window']}{marker}")
        print(f"     Must: {result['must']} | Should: {result['should']}")
        for i, a in enumerate(window):
            primary = " [HIT]" if a.pos == active[hit_idx].pos else ""
            dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
            print(f"     {a.entity[:65]}{dv}{primary}")

    # ====== SUMMARY ======
    print(f"\n{'='*70}")
    print("  SCORE SUMMARY")
    print(f"{'='*70}")
    total_w, total_r = 0, 0
    for r in results:
        print(f"  {r['raw']:>4.1f}/10  (w={r['weight']})  must={r['must']}  should={r['should']}  TF-IDF={r['tfidf']}")
        total_w += r['weighted']
        total_r += r['weight']

    avg = total_w / total_r if total_r > 0 else 0
    print(f"  {'─'*50}")
    print(f"  WEIGHTED AVERAGE: {avg:.1f}/10")

    # ====== DETAILED ANALYSIS ======
    print(f"\n{'='*70}")
    print("  DETAILED ANALYSIS")
    print(f"{'='*70}")

    # Entity quality
    expected_tech = ["Zustand", "Redux", "Vitest", "Playwright", "Webpack", "Vite",
                     "Tailwind", "Radix", "Prisma", "tRPC", "PlanetScale", "Datadog",
                     "Clerk", "Docker", "GitHub Actions", "Vercel", "pnpm", "GraphQL",
                     "Winston", "Storybook", "Cloudflare", "LaunchDarkly", "Flagsmith"]
    found_tech, missing_tech = [], []
    for et in expected_tech:
        found = any(et.lower() in a.entity.lower() for a in active)
        (found_tech if found else missing_tech).append(et)

    print(f"\n  Tech entity capture: {len(found_tech)}/{len(expected_tech)}")
    if missing_tech:
        print(f"  MISSING: {missing_tech}")

    expected_data = ["320KB", "890KB", "2.4MB", "120MB", "800MB", "50MB", "145MB", "680MB",
                     "70%", "4m 12s", "6m 30s", "12ms", "3.2s", "1.8s", "280ms", "4.9",
                     "4.8:1", "44x44", "1000", "5000", "47", "80", "38", "52", "18"]
    found_data, missing_data = [], []
    for ed in expected_data:
        found = any(ed.lower() in a.entity.lower() or (a.data_values and ed.lower() in str(a.data_values).lower()) for a in active)
        (found_data if found else missing_data).append(ed)

    print(f"  Data value capture: {len(found_data)}/{len(expected_data)}")
    if missing_data:
        print(f"  MISSING: {missing_data}")

    # High/low performers
    high = [r for r in results if r["raw"] >= 7]
    low = [r for r in results if r["raw"] < 5]
    print(f"\n  High-scoring (>=7): {len(high)}/{len(results)}")
    print(f"  Low-scoring (<5):  {len(low)}/{len(results)}")

    # Compression
    anchor_chars = sum(len(a.entity) + 15 for a in active)
    ratio = (1 - anchor_chars / total_chars) * 100
    print(f"\n  Compression: {total_chars}→{anchor_chars} chars ({ratio:.1f}% reduction)")
    print(f"  Tokens: ~{total_chars//4}→~{anchor_chars//4}")

    # Verdict
    print(f"\n  VERDICT: {avg:.1f}/10 — {'Good' if avg >= 7 else 'Usable' if avg >= 5 else 'Needs work'}")
    if avg >= 5:
        print("  Core pipeline is functional. Improvements are quality refinements.")
    print()


if __name__ == "__main__":
    main()
