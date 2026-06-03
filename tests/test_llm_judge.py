# -*- coding: utf-8 -*-
"""Test LLM judge with DeepSeek API."""
import sys, os, json
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))
from openai import OpenAI

api_key = os.environ.get('DEEPSEEK_API_KEY', '')
client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com/v1')

candidates = [
    '0. [VERB:DECISION] decided',
    '1. [NOUN:TECH] Redis',
    '2. [NOUN:TECH] SETNX',
    '3. [NOUN:TECH] PostgreSQL [14.2]',
    '4. [NOUN:DATA] 14.2 [14.2]',
    '5. [NOUN:TECH] auth.ts',
    '6. [NOUN:DATA] ERR_005 [ERR_005]',
    '7. [VERB:ANOMALY] race condition [ERR_005]',
    '8. [NOUN:TERM] current',
    '9. [VERB:DECISION] set to [20]',
]

prompt = """Select the 6 most significant anchors from this conversation excerpt.

Conversation excerpt:
We decided to use Redis SETNX for distributed lock. PostgreSQL 14.2 for database. Found race condition bug ERR_005 at auth.ts.

Candidates:
""" + "\n".join(candidates) + """

Return ONLY a JSON array of selected anchors. Each anchor: {"index": <number>, "tags": [<2-4 semantic tags>]}

Rules:
- Prioritize: decisions > anomalies > facts
- Skip: generic verbs (set to), generic nouns (current)
- Prefer anchors with exact data values [in brackets]
- Tags: short semantic categories (e.g. "database", "cache", "auth", "error", "performance")
- Max 4 tags per anchor

Return ONLY the JSON array, no other text."""

r = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0, max_tokens=500,
)
content = r.choices[0].message.content.strip()
print(f"Raw ({r.usage.total_tokens} tokens):")
print(content)
print()

# Parse - handle markdown code blocks
if content.startswith("```"):
    lines = content.split("\n")
    content = "\n".join(lines[1:-1])
    if content.startswith("json"):
        content = content[4:]
content = content.strip()

try:
    selections = json.loads(content)
    print(f"Selected {len(selections)} anchors:")
    for s in selections:
        idx = s["index"]
        tags = s.get("tags", [])
        name = candidates[idx].split("] ", 1)[1] if idx < len(candidates) else f"idx={idx}"
        print(f"  [{idx}] {name:<30s} tags: {tags}")
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Content: {repr(content)}")
