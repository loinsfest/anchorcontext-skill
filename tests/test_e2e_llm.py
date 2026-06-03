"""End-to-end verification with real LLM calls.

Validates that anchor-based reconstruction produces useful context.
Supports Anthropic and DeepSeek (via OpenAI-compatible API).

Set environment variables:
    ANCHOR_TEST_API_KEY    API key
    ANCHOR_TEST_BASE_URL   API base URL (default: https://api.deepseek.com/v1)
    ANCHOR_TEST_MODEL      Model name (default: deepseek-chat)

Usage:
    python tests/test_e2e_llm.py                # With real LLM
    python tests/test_e2e_llm.py --dry-run      # Print prompts only
"""

import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "anchor-context" / "scripts"))

from anchor.extractor import extract_anchors
from anchor.reconstructor import SequenceRetriever

# Sample conversation simulating a real development session
SAMPLE_CONVERSATION = [
    {"content": "我们需要做一个用户认证系统"},
    {"content": "我建议用 JWT token，存在 localStorage 里"},
    {"content": "不对，localStorage 有 XSS 风险，改用 httpOnly cookie"},
    {"content": "决定用 Redis 存 refresh token，SETNX 做分布式锁"},
    {"content": "数据库选 PostgreSQL 14.2，开了连接池 pool size 20"},
    {"content": "auth.ts:42 发现 race condition — JWT 刷新逻辑有并发问题"},
    {"content": "错误码返回 ERR_005，原因是两个请求同时刷新 token"},
    {"content": "修复方案：用 Redis SETNX 加锁，锁超时 5 秒"},
    {"content": "加了 rate limiting，每个 IP 最多 100 req/min"},
    {"content": "部署到生产，API 响应时间从 200ms 降到 80ms"},
]

# Evaluation queries and expected topics
EVAL_QUERIES = [
    ("Redis 锁的方案是什么", ["Redis", "SETNX", "分布式锁"]),
    ("auth.ts 有什么问题", ["auth.ts", "race condition", "ERR_005"]),
    ("数据库用的什么", ["PostgreSQL", "14.2"]),
]


def call_llm(prompt: str, model: str, base_url: str, api_key: str) -> str:
    """Call LLM API (OpenAI-compatible)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai package not installed. Install with: pip install openai")
        return ""

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the anchor context provided. Be concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content


def evaluate_response(response: str, expected_terms: list[str]) -> int:
    """Score response by how many expected terms appear."""
    score = 0
    for term in expected_terms:
        if term.lower() in response.lower():
            score += 1
    # Scale to 0-10
    return min(10, int(score / len(expected_terms) * 10))


def main():
    parser = argparse.ArgumentParser(description="E2E anchor reconstruction test")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling LLM")
    args = parser.parse_args()

    print("=" * 60)
    print(" Anchor Context — E2E LLM Verification")
    print("=" * 60)

    # Step 1: Extract anchors
    print("\n[1/3] Extracting anchors from sample conversation...")
    seq = extract_anchors(SAMPLE_CONVERSATION)
    if not seq.anchors:
        print("  FAIL: No anchors extracted — extraction pipeline is broken")
        sys.exit(1)

    n_active = len(seq.get_active())
    print(f"  OK: {n_active} anchors extracted from {len(SAMPLE_CONVERSATION)} messages")
    for a in seq.get_active():
        data_str = f" [{', '.join(a.data_values)}]" if a.data_values else ""
        print(f"    [{a.pos:4d}] [{a.anchor_type.value:12s}] {a.entity}{data_str}")

    # Step 2: Test retrieval
    print("\n[2/3] Testing position-based retrieval...")
    retriever = SequenceRetriever(seq)

    for query, expected_terms in EVAL_QUERIES:
        seq_idx, hit_idx, score = retriever.find_position(query)
        print(f"  Query: '{query}'")
        print(f"    Hit index: {hit_idx}, Score: {score:.3f}")
        if hit_idx < len(seq.anchors):
            print(f"    Matched: [{seq.anchors[hit_idx].anchor_type.value}] {seq.anchors[hit_idx].entity}")

    # Step 3: LLM reconstruction
    print("\n[3/3] Testing LLM reconstruction...")

    if args.dry_run:
        print("  DRY RUN — printing prompts only\n")
        for query, _ in EVAL_QUERIES:
            prompt = retriever.build_reconstruction_prompt(query)
            print(f"\n{'─'*40}")
            print(f"Query: {query}")
            print(f"{'─'*40}")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("\n  Run without --dry-run to test with real LLM")
        return

    api_key = os.environ.get("ANCHOR_TEST_API_KEY", "")
    if not api_key:
        print("  SKIP: Set ANCHOR_TEST_API_KEY to test with real LLM")
        print("  Run with --dry-run to see prompts")
        return

    base_url = os.environ.get("ANCHOR_TEST_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("ANCHOR_TEST_MODEL", "deepseek-chat")

    print(f"  Using model: {model} at {base_url}")
    total_score = 0

    for query, expected_terms in EVAL_QUERIES:
        prompt = retriever.build_reconstruction_prompt(query)
        response = call_llm(prompt, model, base_url, api_key)

        if not response:
            print(f"  Query '{query}': No response")
            continue

        score = evaluate_response(response, expected_terms)
        total_score += score
        print(f"  Query '{query}': Score {score}/10")
        print(f"    Response: {response[:120]}...")

    avg_score = total_score / len(EVAL_QUERIES) if EVAL_QUERIES else 0
    print(f"\n  Average reconstruction score: {avg_score:.1f}/10")
    if avg_score >= 7:
        print("  VERDICT: PASS — anchors provide useful context reconstruction")
    else:
        print("  VERDICT: NEEDS WORK — anchor quality insufficient for reconstruction")


if __name__ == "__main__":
    main()
