"""Cross-domain verb lexicon for anchor type classification.

150+ Chinese + English verbs organized by AnchorType category.
Uses combined regex for O(n) single-pass segmentation instead of O(n*m) per-verb scans.
"""

from enum import Enum
import re

# ── Verb-to-Anchortype mapping ───────────────────────────────────────────
# Each verb maps to exactly one AnchorType. Verbs are the key;
# the anchor's primary payload is the nearby noun entity.

VERB_MAP: dict[str, str] = {}

# === DECISION verbs ===
# Indicate a choice was made, a direction was set, or an approach was selected.
_DECISION_VERBS_CN = [
    "决定", "改用", "放弃", "用了", "增大", "采用", "选择", "切换", "替换",
    "迁移", "升级", "降级", "回滚", "部署", "发布", "上线", "合并", "拆分",
    "重构", "重写", "优化", "调整", "配置", "设置了", "修改为", "更改为",
    "启用", "禁用", "关闭", "打开了", "添加了", "删除了", "移除了",
    "分配", "预留", "限制为", "设定", "指定",
]

_DECISION_VERBS_EN = [
    "decide", "decided", "chose", "choose", "switch", "switched",
    "replace", "replaced", "migrate", "migrated",
    "upgrade", "upgraded", "downgrade", "downgraded",
    "deploy", "deployed", "release", "released",
    "merge", "merged", "split",
    "refactor", "refactored", "rewrite", "rewrote",
    "optimize", "optimized",
    "configure", "configured", "enable", "enabled", "disable", "disabled",
    "add", "added", "remove", "removed", "delete", "deleted",
    "allocate", "allocated",
    "reserve", "reserved", "limit to", "set to", "change to", "modified to",
    "opted for", "went with", "picked", "settled on",
    "adopt", "adopted", "select", "selected",
]

for v in _DECISION_VERBS_CN + _DECISION_VERBS_EN:
    VERB_MAP[v] = "DECISION"

# === DISCOVERY verbs ===
# Indicate finding, locating, or confirming something.
_DISCOVERY_VERBS_CN = [
    "发现", "定位", "确认", "找到", "排查", "追踪", "调试", "分析出",
    "识别", "检测到", "观察到", "注意到", "意识到", "了解到",
    "看出来", "查出来", "测出来", "重现了", "复现了",
]

_DISCOVERY_VERBS_EN = [
    "discover", "discovered", "find", "found", "locate", "located",
    "identify", "identified", "trace", "traced", "debug", "debugged",
    "diagnose", "diagnosed", "detect", "detected",
    "observe", "observed", "notice", "noticed", "realize", "realized",
    "reproduce", "reproduced", "confirm", "confirmed",
    "pinpoint", "pinpointed", "isolate", "isolated",
    "tracked down", "figured out", "narrowed down",
]

for v in _DISCOVERY_VERBS_CN + _DISCOVERY_VERBS_EN:
    VERB_MAP[v] = "DISCOVERY"

# === ANOMALY verbs ===
# Indicate errors, failures, or unexpected behavior.
_ANOMALY_VERBS_CN = [
    "报错", "失败", "超时", "有问题", "崩溃", "挂了", "卡住", "阻塞",
    "泄漏", "溢出", "死锁", "竞争", "冲突", "异常", "错误",
    "不生效", "没用", "无效", "丢失", "损坏", "乱了",
    "返回 null", "返回空", "500", "404", "502", "503", "401", "403",
    "OOM", "CPU 100", "打满", "占满",
]

_ANOMALY_VERBS_EN = [
    "error", "fail", "timeout", "crash", "hang", "block", "leak",
    "overflow", "deadlock", "race condition", "conflict", "exception",
    "broken", "corrupted", "missing", "null pointer", "segfault",
    "panic", "crashed", "OOM", "thrashing", "bottleneck", "degraded",
    "returned null", "returned empty", "threw",
]

for v in _ANOMALY_VERBS_CN + _ANOMALY_VERBS_EN:
    VERB_MAP[v] = "ANOMALY"

# === CONSTRAINT verbs ===
# Indicate requirements, limitations, or dependencies.
_CONSTRAINT_VERBS_CN = [
    "因为", "必须", "不能", "除非", "前提是", "条件是", "依赖于",
    "要求", "需要", "只能", "最多", "最少", "不超过", "不低于",
    "兼容", "不兼容", "不支持", "限制", "约束",
    "受限于", "取决于", "绑定", "耦合",
]

_CONSTRAINT_VERBS_EN = [
    "because", "must", "cannot", "unless", "prerequisite", "depends on",
    "require", "need to", "only", "at most", "at least", "no more than",
    "compatible", "incompatible", "restricted to", "constrained by",
    "bound to", "coupled with", "limited by",
]

for v in _CONSTRAINT_VERBS_CN + _CONSTRAINT_VERBS_EN:
    VERB_MAP[v] = "CONSTRAINT"


# ── Combined regex for O(n) segmentation ─────────────────────────────────
# Build a single alternation pattern sorted by verb length descending
# so longer matches win over shorter ones (e.g. "tracked down" before "down").

def _build_verb_pattern() -> str:
    verbs = sorted(VERB_MAP.keys(), key=len, reverse=True)
    escaped = [re.escape(v) for v in verbs]
    return r'(' + '|'.join(escaped) + r')'

_VERB_PATTERN = _build_verb_pattern()
_VERB_RE = re.compile(_VERB_PATTERN, re.IGNORECASE)


def segment_text(text: str) -> list[tuple[str, str, int, int]]:
    """Segment text into (verb_text, anchor_type, start, end) tuples.

    Single-pass regex scan — O(n) where n is text length.
    Returns matches sorted by position.
    """
    matches = []
    for m in _VERB_RE.finditer(text):
        verb_text = m.group(0)
        anchor_type = VERB_MAP.get(verb_text.lower(), "FACT")
        matches.append((verb_text, anchor_type, m.start(), m.end()))
    matches.sort(key=lambda x: x[2])
    return matches


def get_anchor_type(verb_text: str) -> str:
    """Look up the AnchorType for a given verb text. Returns 'FACT' if unknown."""
    return VERB_MAP.get(verb_text.lower(), "FACT")


def find_verbs_in_window(text: str, start: int, end: int) -> list[tuple[str, str]]:
    """Find verbs within a character window of text. Returns (verb, type) pairs."""
    window = text[max(0, start):min(len(text), end)]
    result = []
    for m in _VERB_RE.finditer(window):
        verb = m.group(0)
        result.append((verb, VERB_MAP.get(verb.lower(), "FACT")))
    return result
