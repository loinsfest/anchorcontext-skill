# -*- coding: utf-8 -*-
"""Phase 1 diagnostic: trace every anchor through the extraction pipeline.

Instruments extract_anchors to show:
  - Which phase created each anchor
  - What verb triggered it (Phase 2 only)
  - Entity quality classification (signal vs noise)
  - Information density score
"""

import sys, os, pytest
pytest.skip("Legacy diagnostics — not a real test", allow_module_level=True)

# Rest of file is legacy diagnostics code, not run by pytest
import sys, os

from anchor.models import Anchor, AnchorType, EntityClass, AnchorSequence, ENTITY_WEIGHT
from anchor.verbs import segment_text

# Monkey-patch extractor to add tracing
import anchor.extractor as ext

# Conversation from the backend E2E test
import test_long_conversation as t1
messages = t1.CONVERSATION

# ====== Instrumentation ======
trace_log = []       # [(phase, anchor, trigger_verb, quality_label)]
entity_count = {"PHASE1": 0, "PHASE2": 0, "PHASE3": 0}

# Save original Anchor creation
_original_Anchor = ext.Anchor
_original_anchors_list = None
_phase_tracker = 0  # 0 = Phase 1, 1 = Phase 2, 2 = Phase 3

def _quality_label(a):
    """Classify anchor as SIGNAL or NOISE based on entity characteristics."""
    entity = a.entity
    atype = a.anchor_type.value

    # NOISE patterns
    if len(entity) <= 3 and entity.isalpha():
        return "NOISE:short-word"
    if entity in ["Decided", "Current", "Store", "Switched", "Discovered",
                  "Login", "Memory", "Database", "Testing", "Accessibility",
                  "Performance", "Render", "Caught", "Safety"]:
        return "NOISE:common-word"
    if entity.isdigit():
        return "NOISE:bare-number"
    if " + " in entity:
        return "NOISE:compound-cluster"
    if entity.startswith("Specifically ") or entity.startswith("Fixed "):
        return "NOISE:adverb-phrase"
    if entity.endswith("_total") or entity.endswith("_seconds"):
        return "NOISE:metric-suffix"

    # SIGNAL patterns
    if atype == "DECISION" and len(entity) >= 4 and not entity[0].islower():
        return "SIGNAL:decision"
    if atype == "ANOMALY" and len(entity) >= 3:
        return "SIGNAL:anomaly"
    if atype == "CONSTRAINT" and len(entity) >= 4:
        return "SIGNAL:constraint"
    if atype == "DISCOVERY" and len(entity) >= 4:
        return "SIGNAL:discovery"
    if a.entity_class == EntityClass.DATA and (a.data_values or len(entity) > 5):
        return "SIGNAL:data"

    return "NOISE:low-density"


# Run extraction with tracing
seq = ext.extract_anchors(messages)

# Classify all anchors
active = seq.get_active()
signals = []
noises = []
for a in active:
    label = _quality_label(a)
    if label.startswith("SIGNAL"):
        signals.append((label, a))
    else:
        noises.append((label, a))

# ====== REPORT ======
print("=" * 70)
print("  PHASE 1: DIAGNOSTIC EVIDENCE")
print("=" * 70)
print(f"  Input: {len(messages)} msgs, {sum(len(m['content']) for m in messages)} chars")
print(f"  Total anchors: {len(active)}")
print(f"  SIGNAL anchors: {len(signals)} ({len(signals)/max(1,len(active))*100:.0f}%)")
print(f"  NOISE anchors:  {len(noises)} ({len(noises)/max(1,len(active))*100:.0f}%)")
print()

# Signal breakdown
print("  === SIGNAL ANCHORS (should keep) ===")
for label, a in sorted(signals, key=lambda x: x[1].pos):
    dv = f" [{', '.join(a.data_values)}]" if a.data_values else ""
    print(f"  [{a.anchor_type.value:12s}] {a.entity[:55]}{dv}  ({label})")

print()
print("  === NOISE ANCHORS (should filter) ===")
noise_counts = {}
for label, a in noises:
    noise_counts[label] = noise_counts.get(label, 0) + 1

print("  Noise breakdown:")
for label, count in sorted(noise_counts.items(), key=lambda x: -x[1]):
    print(f"    {label}: {count}")

# Show a few noise examples
print()
print("  Noise examples:")
for label, a in sorted(noises, key=lambda x: x[1].pos)[:8]:
    print(f"    [{a.anchor_type.value}] '{a.entity}'  ({label})")

# Compression analysis
ideal_count = len(signals)
ideal_chars = sum(len(a.entity) + 15 for a in signals)
total_chars = sum(len(m["content"]) for m in messages)
current_chars = sum(len(a.entity) + 15 for a in active)

print()
print("  === COMPRESSION IMPACT ===")
print(f"  Original:      {total_chars} chars (~{total_chars//4} tokens)")
print(f"  Current (all): {current_chars} chars (~{current_chars//4} tokens) — {100-current_chars*100//total_chars}% reduction")
print(f"  Ideal (signal):{ideal_chars} chars (~{ideal_chars//4} tokens) — {100-ideal_chars*100//total_chars}% reduction")
print(f"  Noise removed: {len(noises)} anchors, {current_chars-ideal_chars} chars wasted")

# By anchor type
from collections import Counter
print()
print("  === ANCHOR TYPE DISTRIBUTION ===")
for atype in Counter(a.anchor_type.value for a in active).most_common():
    signal_ct = sum(1 for a in active if a.anchor_type.value == atype and _quality_label(a).startswith("SIGNAL"))
    noise_ct = sum(1 for a in active if a.anchor_type.value == atype and not _quality_label(a).startswith("SIGNAL"))
    print(f"    {atype}: {signal_ct} signal + {noise_ct} noise = {signal_ct+noise_ct} total")

print()
print(f"  KEY FINDING: {len(noises)}/{len(active)} anchors ({len(noises)/len(active)*100:.0f}%) are noise")
print(f"  If filtered: compression improves from {100-current_chars*100//total_chars}% to {100-ideal_chars*100//total_chars}% reduction")
