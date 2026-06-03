"""LLM-based significance judge — replaces all hand-crafted rules.

Single Haiku/DeepSeek call per compaction to select the most important
verb-noun pairs and generate semantic tags. Removes the need for
dictionaries, scoring formulas, quotas, and filters.
"""

import json
import os
import sys
from typing import Optional


# Default to DeepSeek (cheapest), override with env vars
_DEFAULT_MODEL = os.environ.get("ANCHOR_JUDGE_MODEL", "deepseek-chat")
_DEFAULT_BASE_URL = os.environ.get("ANCHOR_JUDGE_BASE_URL", "https://api.deepseek.com/v1")
_DEFAULT_API_KEY = os.environ.get("ANCHOR_JUDGE_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))


def judge_significance(
    candidates: list[dict],
    conversation_excerpt: str,
    target_count: int = 15,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Ask LLM to select the most significant anchors from candidates.

    Args:
        candidates: List of {"entity": str, "type": "verb"|"noun",
                    "verb_type"|"noun_class": str, "pos": int, "data": [str]}
        conversation_excerpt: First ~2000 chars of conversation for context
        target_count: How many anchors to select (default 15)
        model: LLM model name
        base_url: API base URL (OpenAI-compatible)
        api_key: API key

    Returns:
        List of selected candidates with tags added:
        {"entity": str, "type": "verb"|"noun", "tags": [str]}
    """
    model = model or _DEFAULT_MODEL
    base_url = base_url or _DEFAULT_BASE_URL
    api_key = api_key or _DEFAULT_API_KEY

    if not api_key:
        return _fallback_select(candidates, target_count)

    if len(candidates) <= target_count:
        # No LLM needed — just add empty tags and return all
        return [{"entity": c["entity"], "type": c["type"], "tags": []} for c in candidates]

    try:
        return _call_llm(candidates, conversation_excerpt, target_count, model, base_url, api_key)
    except Exception as e:
        print(f"[anchor-judge] LLM call failed: {e}", file=sys.stderr)
        return _fallback_select(candidates, target_count)


def _call_llm(candidates, excerpt, target, model, base_url, api_key):
    """Make the LLM API call."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[anchor-judge] openai package not installed. Install: pip install openai", file=sys.stderr)
        return _fallback_select(candidates, target)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build compact prompt listing all candidates
    candidate_list = []
    for i, c in enumerate(candidates):
        data_str = f" [{', '.join(c['data'])}]" if c.get("data") else ""
        if c["type"] == "verb":
            candidate_list.append(f"{i}. [VERB:{c.get('verb_type','?')}] {c['entity']}{data_str}")
        else:
            candidate_list.append(f"{i}. [NOUN:{c.get('noun_class','?')}] {c['entity']}{data_str}")

    prompt = f"""Select the {target} most significant anchors from this conversation excerpt.

Conversation excerpt:
{excerpt[:2000]}

Candidates ({len(candidate_list)} verb-noun pairs):
{chr(10).join(candidate_list)}

Return ONLY a JSON array of selected anchors. Each anchor: {{"index": <candidate_number>, "tags": [<2-4 semantic category tags>]}}.

Rules:
- Prioritize: decisions > anomalies > discoveries > constraints > facts
- Prefer anchors with exact data values (numbers, error codes, versions)
- Skip: generic verbs (set to, add, use), generic nouns (min, UTC, percent)
- Tags: semantic categories in English (e.g. "database", "cache", "auth", "performance")
- Maximum 4 tags per anchor, each 1-2 words

Return ONLY the JSON array, no other text."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise ValueError("LLM returned empty response")

    # Parse JSON response — handle markdown code blocks
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) >= 2 else content
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    selections = json.loads(content)

    # Map back to candidates
    result = []
    for sel in selections[:target]:
        idx = sel.get("index", -1)
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            result.append({
                "entity": c["entity"],
                "type": c["type"],
                "tags": sel.get("tags", []),
            })

    return result


def _fallback_select(candidates, target):
    """Zero-cost fallback: balanced scoring when LLM is unavailable.

    Uses the old quota approach to ensure verb/noun/semantic diversity.
    No API key needed — same behavior as the previous regex-only version.
    """
    # Separate by type
    verbs = [c for c in candidates if c["type"] == "verb"]
    nouns_data = [c for c in candidates if c["type"] == "noun" and c.get("noun_class") == "DATA"]
    nouns_tech = [c for c in candidates if c["type"] == "noun" and c.get("noun_class") != "DATA"]

    # Score within each type
    def score_verb(c):
        p = {"DECISION": 5, "DISCOVERY": 4, "ANOMALY": 3, "CONSTRAINT": 2, "FACT": 1}
        return p.get(c.get("verb_type", "FACT"), 1) + (2 if c.get("data") else 0)

    def score_noun(c):
        w = {"DATA": 1.0, "TECH": 0.7, "TERM": 0.5}
        s = w.get(c.get("noun_class", "TERM"), 0.5) * 5
        s += 2 if c.get("data") else 0
        s += 1 if '.' in c.get("entity", "") else 0
        s += 1 if ' ' in c.get("entity", "") else 0  # Multi-word = more specific
        s += 0.5 * (len(c.get("entity", "")) / 10)    # Longer = more specific
        return s

    verbs.sort(key=score_verb, reverse=True)
    nouns_data.sort(key=score_noun, reverse=True)
    nouns_tech.sort(key=score_noun, reverse=True)

    # Quota: at least 2 of each type (deduplicated by entity)
    selected = []
    seen = set()

    def add_unique(c):
        key = (c["entity"], c["type"])
        if key not in seen:
            seen.add(key)
            selected.append(c)

    for c in verbs[:2]:
        add_unique(c)
    for c in nouns_tech[:3]:
        add_unique(c)
    for c in nouns_data[:3]:
        add_unique(c)

    # Fill remaining by score across all types
    remaining_verbs = verbs[2:]
    remaining_nouns = nouns_tech[3:] + nouns_data[3:]
    all_remaining = [(score_verb(c), c) for c in remaining_verbs]
    all_remaining += [(score_noun(c), c) for c in remaining_nouns]
    all_remaining.sort(key=lambda x: x[0], reverse=True)

    for _, c in all_remaining:
        if len(selected) >= target:
            break
        add_unique(c)

    return [{"entity": c["entity"], "type": c["type"], "tags": []} for c in selected[:target]]
