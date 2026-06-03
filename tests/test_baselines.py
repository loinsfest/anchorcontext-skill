# -*- coding: utf-8 -*-
"""Benchmark: anchor-context vs traditional compression baselines.

Tests 4 methods on the same 30-msg backend conversation and 40-msg frontend
conversation, comparing reconstruction fidelity and compression ratio.
"""

import sys, os, random
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))

from anchor.extractor import extract_anchors
from anchor.reconstructor import SequenceRetriever

# Import test data from both conversations
from test_long_conversation import CONVERSATION as BACKEND_CONV, GROUND_TRUTH as BACKEND_GT
from test_long_conversation2 import CONVERSATION as FRONTEND_CONV, GROUND_TRUTH as FRONTEND_GT


# ============================================================
# METHOD 1: Sliding Window — keep last N% of messages
# ============================================================
class SlidingWindow:
    def __init__(self, messages, keep_ratio=0.25):
        keep_n = max(1, int(len(messages) * keep_ratio))
        self.kept = messages[-keep_n:]

    def retrieve(self, query):
        return " ".join(m["content"] for m in self.kept)

    @property
    def tokens(self):
        return sum(len(m["content"]) for m in self.kept) // 4


# ============================================================
# METHOD 2: Extractive Keyword — keep sentences with key terms
# ============================================================
class ExtractiveKeywords:
    def __init__(self, messages):
        # Build keyword index from the FULL conversation
        import re
        self.keyword_sentences = []
        for m in messages:
            # Split into sentences
            sentences = re.split(r'[.!?]\s+', m["content"])
            for sent in sentences:
                if len(sent) > 15:  # Skip very short fragments
                    self.keyword_sentences.append(sent.strip())

    def retrieve(self, query):
        # Return sentences containing ANY query word
        query_words = set(query.lower().split())
        matching = [s for s in self.keyword_sentences
                    if any(w in s.lower() for w in query_words)]
        return " ".join(matching[:10])

    @property
    def tokens(self):
        return sum(len(s) for s in self.keyword_sentences) // 4


# ============================================================
# METHOD 3: Claude-style Compaction Simulation
# Keeps: first 2 msgs (task setup), last 5 msgs (recent),
#        and top-5 sentences by keyword density
# ============================================================
class CompactionSim:
    def __init__(self, messages):
        import re
        self.first = messages[:2]
        self.last = messages[-5:]
        # Pick the 5 most keyword-dense sentences from the middle
        middle = messages[2:-5]
        all_sentences = []
        for m in middle:
            for sent in re.split(r'[.!?]\s+', m["content"]):
                if len(sent) > 30:
                    all_sentences.append(sent.strip())
        # Score by keyword density (proper nouns, numbers)
        scored = []
        for s in all_sentences:
            score = len(re.findall(r'[A-Z][a-z]{2,}|\d+\.?\d*', s))
            scored.append((score, s))
        scored.sort(reverse=True)
        self.key_sentences = [s for _, s in scored[:5]]

    def retrieve(self, query):
        parts = [m["content"] for m in self.first]
        parts.extend(self.key_sentences)
        parts.extend(m["content"] for m in self.last)
        return " ".join(parts)

    @property
    def tokens(self):
        total = sum(len(m["content"]) for m in self.first)
        total += sum(len(s) for s in self.key_sentences)
        total += sum(len(m["content"]) for m in self.last)
        return total // 4


# ============================================================
# METHOD 4: Anchor Context (ours)
# ============================================================
class AnchorMethod:
    def __init__(self, messages):
        self.seq = extract_anchors(messages)
        self.retriever = SequenceRetriever(self.seq)

    def retrieve(self, query):
        _, hit_idx, _ = self.retriever.find_position(query)
        window = self.retriever.get_window(hit_idx, radius=2)
        text = " ".join(a.entity for a in window)
        for a in window:
            if a.data_values:
                text += " " + " ".join(a.data_values)
        return text

    @property
    def tokens(self):
        return sum(len(a.entity) + 15 for a in self.seq.get_active()) // 4


# ============================================================
# Scoring (same as E2E tests)
# ============================================================
def score(text, gt):
    must_hits = sum(1 for t in gt["must_contain"] if t.lower() in text.lower())
    should_hits = sum(1 for t in gt["should_contain"] if t.lower() in text.lower())
    raw = min(10, (must_hits / len(gt["must_contain"])) * 10 +
              (should_hits / max(1, len(gt["should_contain"]))) * 2)
    return {
        "must": f"{must_hits}/{len(gt['must_contain'])}",
        "should": f"{should_hits}/{len(gt['should_contain'])}",
        "raw": round(raw, 1),
    }


def benchmark(name, messages, ground_truth, method_class, *args):
    """Run a full benchmark for one method."""
    method = method_class(messages, *args) if args else method_class(messages)
    total_raw = 0
    total_w = 0
    results = []

    for query, gt in ground_truth.items():
        text = method.retrieve(query)
        s = score(text, gt)
        s["query"] = query[:50]
        s["weighted"] = round(s["raw"] * gt["score_weight"], 1)
        results.append(s)
        total_raw += s["raw"]
        total_w += gt["score_weight"]

    avg = total_raw / len(ground_truth)
    wavg = sum(r["weighted"] for r in results) / total_w if total_w > 0 else 0
    tokens = method.tokens
    return results, avg, wavg, tokens


def print_results(label, results, avg, wavg, tokens, original_tokens):
    comp = (1 - tokens / original_tokens) * 100
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    for r in results:
        print(f"    {r['raw']:>4.1f}  {r['query'][:45]:<45s}  M:{r['must']} S:{r['should']}")
    print(f"  {'─'*60}")
    print(f"  Avg: {avg:.1f}/10  Weighted: {wavg:.1f}/10  "
          f"Tokens: {tokens}  Compression: {comp:.1f}%")


def main():
    random.seed(42)

    for dataset_name, messages, ground_truth in [
        ("BACKEND (30 msgs, ~918 tokens)", BACKEND_CONV, BACKEND_GT),
        ("FRONTEND (40 msgs, ~1813 tokens)", FRONTEND_CONV, FRONTEND_GT),
    ]:
        original_tokens = sum(len(m["content"]) for m in messages) // 4

        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {dataset_name}")
        print(f"{'='*70}")

        # Raw full conversation (upper bound)
        full_text = " ".join(m["content"] for m in messages)
        total_r = 0
        total_w = 0
        full_results = []
        for q, gt in ground_truth.items():
            s = score(full_text, q, gt)
            s["query"] = q[:50]
            s["weighted"] = round(s["raw"] * gt["score_weight"], 1)
            full_results.append(s)
            total_r += s["raw"]
            total_w += gt["score_weight"]
        full_avg = total_r / len(ground_truth)
        full_wavg = sum(r["weighted"] for r in full_results) / total_w
        print_results("FULL CONVERSATION (upper bound)", full_results,
                      full_avg, full_wavg, original_tokens, original_tokens)

        # Method 1: Sliding Window
        r, a, w, t = benchmark("SLIDING WINDOW", messages, ground_truth, SlidingWindow, 0.25)
        print_results("SLIDING WINDOW (keep last 25%)", r, a, w, t, original_tokens)

        # Method 2: Extractive Keywords
        r, a, w, t = benchmark("EXTRACTIVE KEYWORDS", messages, ground_truth, ExtractiveKeywords)
        print_results("EXTRACTIVE KEYWORDS (keyword match)", r, a, w, t, original_tokens)

        # Method 3: Compaction Sim
        r, a, w, t = benchmark("COMPACTION SIM (Claude-style)", messages, ground_truth, CompactionSim)
        print_results("COMPACTION SIM (first+last+key sentences)", r, a, w, t, original_tokens)

        # Method 4: Anchor Context
        r, a, w, t = benchmark("ANCHOR CONTEXT", messages, ground_truth, AnchorMethod)
        print_results("ANCHOR CONTEXT (ours)", r, a, w, t, original_tokens)

    # ====== SUMMARY TABLE ======
    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Method':<35s} {'Backend':>8s} {'Frontend':>9s} {'Avg Compression':>15s}")
    print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*15}")
    # We'll fill these in manually after both datasets run
    print()


if __name__ == "__main__":
    main()
