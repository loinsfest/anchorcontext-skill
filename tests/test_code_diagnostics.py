# -*- coding: utf-8 -*-
"""Phase 1: Diagnostic — trace code block contamination in extraction."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))
from anchor.extractor import extract_graph, _extract_entities, segment_text

# Conversation WITH code blocks — realistic Claude Code session
msgs = [
    {'content': 'We decided to use Redis SETNX for distributed locking. Here is the implementation:'},
    {'content': '''```python
def acquire_lock(redis_client, lock_key, timeout=5):
    try:
        return redis_client.set(lock_key, 'locked', nx=True, ex=timeout)
    except RedisError as e:
        log_error(f"Lock acquisition failed: {e}")
        return False

class LockManager:
    def __init__(self, redis_url: str):
        self.client = Redis.from_url(redis_url)

    def with_lock(self, key: str, fn):
        if acquire_lock(self.client, key):
            try:
                return fn()
            finally:
                self.client.delete(key)
```'''},
    {'content': 'This caused a race condition in auth.ts at line 42. Error code ERR_005.'},
    {'content': 'We also added rate limiting. Here is the nginx config:'},
    {'content': '''```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
server {
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://auth_service:8080;
    }
}
```'''},
    {'content': 'Database migration added a new column:'},
    {'content': '''```sql
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
ALTER TABLE users ADD COLUMN login_count INT DEFAULT 0;
CREATE INDEX idx_users_last_login ON users(last_login);
```'''},
    {'content': 'Post-deployment: API latency dropped from 200ms to 80ms. Memory stable at 120MB.'},
]

full_text = '\n'.join(m['content'] for m in msgs)
total_chars = len(full_text)

print("=" * 70)
print("PHASE 1: Diagnostic Evidence — Code Block Contamination")
print("=" * 70)
print(f"Input: {len(msgs)} messages, {total_chars} chars")
print()

# Count code blocks
import re
code_blocks = re.findall(r'```[\s\S]*?```', full_text)
print(f"Code blocks detected: {len(code_blocks)}")
code_chars = sum(len(b) for b in code_blocks)
print(f"Code chars: {code_chars} ({code_chars/total_chars*100:.0f}% of total)")
print()

# Verbs in code vs text
print("=== Verbs found ===")
verbs = segment_text(full_text)
in_code = 0
for v, t, s, e in verbs:
    in_block = any(s >= full_text.find(b) and s < full_text.find(b) + len(b) for b in code_blocks)
    if in_block:
        in_code += 1
        print(f"  [IN CODE] pos={s:4d} [{t:12s}] {v}")
    else:
        print(f"  [IN TEXT] pos={s:4d} [{t:12s}] {v}")
print(f"  Verbs in code: {in_code}/{len(verbs)}")

print()
print("=== Entities found ===")
entities = _extract_entities(full_text)
in_code_n = 0
for e, ec, s, en in entities:
    in_block = any(s >= full_text.find(b) and s < full_text.find(b) + len(b) for b in code_blocks)
    if in_block:
        in_code_n += 1
        print(f"  [IN CODE] pos={s:4d} [{ec.value:5s}] {e!r}")
    else:
        print(f"  [IN TEXT] pos={s:4d} [{ec.value:5s}] {e!r}")
print(f"  Entities in code: {in_code_n}/{len(entities)}")

print()
g = extract_graph(msgs)
noise = []
signal = []
for n in g.noun_anchors:
    entity_src = full_text[max(0, n.pos-5):n.pos+len(n.entity)+5]
    in_block = any(n.pos >= full_text.find(b) and n.pos < full_text.find(b) + len(b) for b in code_blocks)
    if in_block:
        noise.append(n)
    else:
        signal.append(n)

print(f"=== Anchors: {g.total_anchors} ({len(g.verb_anchors)}v+{len(g.noun_anchors)}n) ===")
print(f"  From code blocks (noise): {len(noise)}")
for n in noise:
    print(f"    [NOISE] {n.entity!r}")
print(f"  From text (signal): {len(signal)}")
for n in signal:
    print(f"    [SIGNAL] {n.entity!r}")
