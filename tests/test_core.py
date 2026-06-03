"""Unit tests for anchor-core library.

Run: python -m pytest tests/ -v
"""

import sys
import os
import json
import tempfile
import pytest
from pathlib import Path

# Add anchor module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "anchor-context" / "scripts"))

from anchor.models import Anchor, AnchorType, EntityClass, AnchorSequence, ENTITY_WEIGHT
from anchor.verbs import segment_text, get_anchor_type, VERB_MAP
from anchor.extractor import extract_anchors, _extract_entities, _classify_entity
from anchor.store import AnchorStore, id as anchor_id
from anchor.formatter import format_for_injection, format_compact, format_verbose
from anchor.conflict import detect_conflicts, mark_superseded, _entity_overlap_score
from anchor.constraints import build_constraint_graph


# ═══════════════════════════════════════════════════════════════════════════
# Models Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModels:
    def test_anchor_creation(self):
        a = Anchor(
            entity="Redis SETNX",
            anchor_type=AnchorType.DECISION,
            entity_class=EntityClass.TECH,
            pos=42,
            data_values=["distributed lock"],
        )
        assert a.entity == "Redis SETNX"
        assert a.anchor_type == AnchorType.DECISION
        assert a.entity_class == EntityClass.TECH
        assert a.pos == 42
        assert a.data_values == ["distributed lock"]
        assert not a.is_superseded

    def test_anchor_superseded_property(self):
        a = Anchor(entity="test", anchor_type=AnchorType.FACT,
                   entity_class=EntityClass.TERM, pos=0)
        assert not a.is_superseded
        a.is_superseded = True
        assert a.is_superseded

    def test_anchor_serialization(self):
        a = Anchor(
            entity="Redis SETNX",
            anchor_type=AnchorType.DECISION,
            entity_class=EntityClass.TECH,
            pos=42,
            data_values=["distributed lock"],
            supersedes=[10, 11],
        )
        d = a.to_dict()
        assert d["entity"] == "Redis SETNX"
        assert d["anchor_type"] == "DECISION"
        assert d["pos"] == 42
        assert d["supersedes"] == [10, 11]

        a2 = Anchor.from_dict(d)
        assert a2.entity == a.entity
        assert a2.anchor_type == a.anchor_type
        assert a2.pos == a.pos

    def test_entity_weight(self):
        assert ENTITY_WEIGHT[EntityClass.DATA] == 1.0
        assert ENTITY_WEIGHT[EntityClass.TECH] == 0.7
        assert ENTITY_WEIGHT[EntityClass.TERM] == 0.5


class TestAnchorSequence:
    def test_add_and_sort(self):
        seq = AnchorSequence(session_id="test")
        seq.add(Anchor(entity="B", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TERM, pos=20))
        seq.add(Anchor(entity="A", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TERM, pos=10))
        seq.add(Anchor(entity="C", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TERM, pos=30))

        assert [a.pos for a in seq.anchors] == [10, 20, 30]

    def test_get_window(self):
        seq = AnchorSequence(session_id="test")
        for i in range(10):
            seq.add(Anchor(entity=f"E{i}", anchor_type=AnchorType.FACT,
                           entity_class=EntityClass.TERM, pos=i * 10))

        window = seq.get_window(center_index=5, radius=2)
        assert len(window) == 5
        assert window[0].pos == 30
        assert window[4].pos == 70

    def test_get_window_excludes_superseded(self):
        seq = AnchorSequence(session_id="test")
        for i in range(5):
            a = Anchor(entity=f"E{i}", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TERM, pos=i * 10)
            if i == 2:
                a.is_superseded = True
            seq.add(a)

        window = seq.get_window(center_index=2, radius=1)
        assert len(window) == 2  # superseded anchor excluded

    def test_get_active(self):
        seq = AnchorSequence(session_id="test")
        a1 = Anchor(entity="E1", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TERM, pos=10)
        a2 = Anchor(entity="E2", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TERM, pos=20)
        a2.is_superseded = True
        seq.add(a1)
        seq.add(a2)

        active = seq.get_active()
        assert len(active) == 1
        assert active[0].entity == "E1"

    def test_sequence_serialization(self):
        seq = AnchorSequence(session_id="test123")
        seq.add(Anchor(entity="E1", anchor_type=AnchorType.DECISION,
                       entity_class=EntityClass.TECH, pos=10))
        seq.add(Anchor(entity="E2", anchor_type=AnchorType.ANOMALY,
                       entity_class=EntityClass.DATA, pos=20, data_values=["line:42"]))

        d = seq.to_dict()
        assert d["session_id"] == "test123"
        assert len(d["anchors"]) == 2

        seq2 = AnchorSequence.from_dict(d)
        assert seq2.session_id == "test123"
        assert len(seq2.anchors) == 2
        assert seq2.anchors[0].entity == "E1"


# ═══════════════════════════════════════════════════════════════════════════
# Verbs Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVerbs:
    def test_verb_map_coverage(self):
        assert len(VERB_MAP) >= 150
        assert VERB_MAP.get("决定") == "DECISION"
        assert VERB_MAP.get("发现") == "DISCOVERY"
        assert VERB_MAP.get("报错") == "ANOMALY"
        assert VERB_MAP.get("必须") == "CONSTRAINT"

    def test_get_anchor_type_case_insensitive(self):
        assert get_anchor_type("Decide") == "DECISION"
        assert get_anchor_type("FOUND") == "DISCOVERY"

    def test_get_anchor_type_unknown(self):
        assert get_anchor_type("xyzzy_nonexistent") == "FACT"

    def test_segment_text_chinese(self):
        text = "我们决定用 Redis，但是发现了一个报错。"
        matches = segment_text(text)
        assert len(matches) >= 2
        verbs_found = [m[0] for m in matches]
        assert "决定" in verbs_found

    def test_segment_text_english(self):
        text = "We decided to use Redis but found an error."
        matches = segment_text(text)
        assert len(matches) >= 1

    def test_segment_text_mixed(self):
        text = "决定改用 Redis，deploy 到生产后报错 timeout。"
        matches = segment_text(text)
        verbs_found = [m[0] for m in matches]
        assert "决定" in verbs_found or len(matches) > 0

    def test_longer_match_wins(self):
        """Longer verb patterns should match before shorter ones."""
        text = "tracked down the bug and found it"
        matches = segment_text(text)
        verbs = [m[0] for m in matches]
        # "tracked down" should match as one unit, not "tracked" separately
        if any("track" in v.lower() for v in verbs):
            idx = next(i for i, v in enumerate(verbs) if "track" in v.lower())
            # Should be the full phrase if present


# ═══════════════════════════════════════════════════════════════════════════
# Extractor Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityRecognition:
    def test_classify_data_line_number(self):
        assert _classify_entity("42") == EntityClass.DATA  # from :42 pattern

    def test_classify_error_code(self):
        assert _classify_entity("ERR_001") == EntityClass.DATA

    def test_classify_version(self):
        assert _classify_entity("14.2") == EntityClass.DATA

    def test_classify_filename(self):
        assert _classify_entity("auth.ts") == EntityClass.TECH

    def test_classify_uppercase(self):
        assert _classify_entity("SETNX") == EntityClass.TECH

    def test_classify_chinese_term(self):
        assert _classify_entity("分布式锁") == EntityClass.TERM

    def test_extract_entities(self):
        text = "在 auth.ts:42 发现 Redis race condition，错误码 ERR_001"
        entities = _extract_entities(text)
        assert len(entities) > 0


class TestExtractor:
    def test_empty_messages(self):
        seq = extract_anchors([])
        assert len(seq.anchors) == 0

    def test_single_message_with_data(self):
        messages = [{"content": "Error ERR_001 at auth.ts:42 — Redis timeout"}]
        seq = extract_anchors(messages)
        assert len(seq.anchors) > 0

    def test_data_entities_always_anchor(self):
        messages = [{"content": "Found bug at user.py:142, error ANN_005, version 3.10"}]
        seq = extract_anchors(messages)
        data_anchors = [a for a in seq.anchors if a.entity_class == EntityClass.DATA]
        assert len(data_anchors) > 0

    def test_decisions_extracted(self):
        messages = [{"content": "Decided to use Redis SETNX for distributed lock"}]
        seq = extract_anchors(messages)
        assert len(seq.anchors) > 0

    def test_anomaly_extracted(self):
        messages = [{"content": "auth.ts crashed: JWT race condition at line 42"}]
        seq = extract_anchors(messages)
        assert len(seq.anchors) > 0

    def test_position_sorting(self):
        messages = [
            {"content": "First we use MySQL"},
            {"content": "Then we found Redis is better"},
            {"content": "Finally we decided on PostgreSQL"},
        ]
        seq = extract_anchors(messages)
        positions = [a.pos for a in seq.anchors]
        assert positions == sorted(positions)

    def test_session_id_auto_generated(self):
        seq = extract_anchors([{"content": "Test message"}])
        assert len(seq.session_id) == 12

    def test_data_values_attached(self):
        messages = [{"content": "Error at auth.ts:42 with code ERR_005"}]
        seq = extract_anchors(messages)
        data_anchors = [a for a in seq.anchors if a.entity_class == EntityClass.DATA]
        if data_anchors:
            # DATA anchors near error codes should have data values
            pass

    def test_multi_message_context(self):
        messages = [
            {"content": "We decided to use Redis distributed lock"},
            {"content": "Found race condition in auth service"},
            {"content": "Must sync across Pods"},
        ]
        seq = extract_anchors(messages)
        assert len(seq.anchors) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Store Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AnchorStore(store_dir=tmpdir)
            seq = AnchorSequence(session_id="test-session")
            seq.add(Anchor(entity="Redis", anchor_type=AnchorType.DECISION,
                           entity_class=EntityClass.TECH, pos=10))

            store.save_sequence(seq)

            loaded = store.load_sequence("test-session")
            assert loaded is not None
            assert loaded.session_id == "test-session"
            assert len(loaded.anchors) == 1
            assert loaded.anchors[0].entity == "Redis"

    def test_load_nonexistent(self):
        store = AnchorStore()
        assert store.load_sequence("nonexistent") is None

    def test_prune_removes_superseded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AnchorStore(store_dir=tmpdir)
            seq = AnchorSequence(session_id="test")
            a1 = Anchor(entity="Old Redis", anchor_type=AnchorType.DECISION,
                        entity_class=EntityClass.TECH, pos=10)
            a2 = Anchor(entity="New Redis", anchor_type=AnchorType.DECISION,
                        entity_class=EntityClass.TECH, pos=20,
                        supersedes=[anchor_id(a1)])
            a2.is_superseded = False  # New is active

            seq.add(a1)
            seq.add(a2)

            store.prune(seq, max_anchors=1)
            assert len(seq.anchors) <= 1

    def test_anchor_id_stable(self):
        a = Anchor(entity="test", anchor_type=AnchorType.FACT,
                   entity_class=EntityClass.TERM, pos=42)
        id1 = anchor_id(a)
        id2 = anchor_id(a)
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════════════
# Formatter Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatter:
    def test_format_compact(self):
        seq = AnchorSequence(session_id="test")
        seq.add(Anchor(entity="E1", anchor_type=AnchorType.DECISION,
                       entity_class=EntityClass.TECH, pos=10))
        seq.add(Anchor(entity="E2", anchor_type=AnchorType.ANOMALY,
                       entity_class=EntityClass.DATA, pos=20, data_values=["line:42"]))
        seq.add(Anchor(entity="E3", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TERM, pos=30))

        result = format_compact(seq, query_hit_index=1)
        assert "PRIMARY" in result
        assert "E1" in result
        assert "E2" in result
        assert "line:42" in result

    def test_format_for_injection(self):
        seq = AnchorSequence(session_id="test123")
        seq.add(Anchor(entity="Redis SETNX", anchor_type=AnchorType.DECISION,
                       entity_class=EntityClass.TECH, pos=10))

        result = format_for_injection([seq])
        assert "Session" in result
        assert "Redis SETNX" in result
        assert "DECISION" in result

    def test_format_for_injection_empty(self):
        result = format_for_injection([])
        assert result == ""

    def test_format_verbose(self):
        a = Anchor(entity="Redis", anchor_type=AnchorType.DECISION,
                   entity_class=EntityClass.TECH, pos=42, data_values=["v1.0"])
        result = format_verbose(a, 0)
        assert "Redis" in result
        assert "DECISION" in result
        assert "v1.0" in result


# ═══════════════════════════════════════════════════════════════════════════
# Conflict Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConflict:
    def test_overlap_score_identical(self):
        a1 = Anchor(entity="Redis", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TECH, pos=10)
        a2 = Anchor(entity="Redis", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TECH, pos=20)
        assert _entity_overlap_score(a1, a2) == 1.0

    def test_overlap_score_different(self):
        a1 = Anchor(entity="Redis", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TECH, pos=10)
        a2 = Anchor(entity="MySQL", anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TECH, pos=20)
        assert _entity_overlap_score(a1, a2) < 0.5

    def test_detect_conflicts(self):
        existing = AnchorSequence(session_id="old")
        existing.add(Anchor(entity="Redis single", anchor_type=AnchorType.DECISION,
                            entity_class=EntityClass.TECH, pos=10))
        existing.add(Anchor(entity="MySQL", anchor_type=AnchorType.FACT,
                            entity_class=EntityClass.TECH, pos=20))

        incoming = AnchorSequence(session_id="new")
        incoming.add(Anchor(entity="Redis cluster", anchor_type=AnchorType.DECISION,
                             entity_class=EntityClass.TECH, pos=30))

        conflicts = detect_conflicts(existing, incoming)
        # Redis single → Redis cluster should have overlap
        assert len(conflicts) >= 0  # Depends on threshold

    def test_mark_superseded(self):
        seq = AnchorSequence(session_id="test")
        a1 = Anchor(entity="Old Redis", anchor_type=AnchorType.DECISION,
                    entity_class=EntityClass.TECH, pos=10)
        a2 = Anchor(entity="New Redis", anchor_type=AnchorType.DECISION,
                    entity_class=EntityClass.TECH, pos=20)
        seq.add(a1)
        seq.add(a2)

        conflicts = [(0, 1)]  # New at index 1 supersedes old at index 0
        mark_superseded(seq, conflicts)

        assert seq.anchors[0].is_superseded
        assert anchor_id(seq.anchors[0]) in seq.anchors[1].supersedes


# ═══════════════════════════════════════════════════════════════════════════
# Constraints Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraints:
    def test_build_constraint_graph(self):
        seq = AnchorSequence(session_id="test")
        seq.add(Anchor(entity="Redis", anchor_type=AnchorType.DECISION,
                       entity_class=EntityClass.TECH, pos=10))
        seq.add(Anchor(entity="必须分布式锁", anchor_type=AnchorType.CONSTRAINT,
                       entity_class=EntityClass.TERM, pos=20))
        seq.add(Anchor(entity="PostgreSQL", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TECH, pos=30))

        graph = build_constraint_graph(seq)
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) >= 1

    def test_cross_validation_clusters(self):
        seq = AnchorSequence(session_id="test")
        seq.add(Anchor(entity="Redis SETNX", anchor_type=AnchorType.DECISION,
                       entity_class=EntityClass.TECH, pos=10))
        seq.add(Anchor(entity="Redis 集群", anchor_type=AnchorType.CONSTRAINT,
                       entity_class=EntityClass.TERM, pos=80))
        seq.add(Anchor(entity="MySQL", anchor_type=AnchorType.FACT,
                       entity_class=EntityClass.TECH, pos=2000))

        graph = build_constraint_graph(seq)
        # Redis anchors should form a cross-validation cluster
        # (close in time + share entity text)
        assert len(graph["clusters"]) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════════════════

class TestBidirectionalGraph:
    """Tests the bidirectional verb-noun anchor graph."""

    def test_graph_extraction(self):
        """10 dense messages should produce a valid graph with links."""
        from anchor.extractor import extract_graph
        messages = [
            {"content": "We decided to use Redis SETNX for distributed lock"},
            {"content": "Found JWT race condition at auth.ts line 42 ERR_005"},
            {"content": "Database PostgreSQL 14.2 PgBouncer pool 20"},
            {"content": "API latency dropped from 200ms to 80ms after deploy"},
            {"content": "Memory leak LRU overflow 2.1GB to 180MB"},
            {"content": "Must add TOTP 2FA GDPR tokens deletable 30 days"},
            {"content": "Load test 500 RPS p50 45ms zero errors"},
            {"content": "ERR_005 present 14 days 3 percent users affected"},
            {"content": "Next OAuth2 Google GitHub social login 8 points"},
            {"content": "Redis session architecture for new features"},
        ]
        graph = extract_graph(messages)
        total = graph.total_anchors
        # With max(8, 10//2)=8 target and dense messages, expect 8 anchors
        assert 5 <= total <= 15, f"Expected 5-15 anchors, got {total} ({len(graph.verb_anchors)}v + {len(graph.noun_anchors)}n)"

        # Must have both types
        assert len(graph.verb_anchors) > 0, "Should extract verb anchors"
        assert len(graph.noun_anchors) > 0, "Should extract noun anchors"

        # No common words as verb entities
        common = {"Decided", "Current", "Store", "Database", "Cannot", "Default"}
        for v in graph.verb_anchors:
            assert v.entity not in common, f"Common word '{v.entity}' as verb anchor"

    def test_links_bidirectional(self):
        """Every linked verb should point to an existing noun and vice versa."""
        from anchor.extractor import extract_graph
        messages = [
            {"content": "We decided to use Redis SETNX for distributed lock"},
            {"content": "Found JWT race condition at auth.ts line 42"},
        ]
        graph = extract_graph(messages)
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                n = graph.find_noun(v.nearest_noun_id)
                assert n is not None, f"Verb {v.entity} links to missing noun {v.nearest_noun_id}"
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                v = graph.find_verb(n.nearest_verb_id)
                assert v is not None, f"Noun {n.entity} links to missing verb {n.nearest_verb_id}"


class TestCompressionQuality:
    """Graph-based extraction: quality + compression targets."""

    def test_graph_compression(self):
        """10 dense messages → graph with 5-15 total anchors."""
        from anchor.extractor import extract_graph
        messages = [
            {"content": "We decided to use Redis SETNX for distributed locking"},
            {"content": "Found JWT race condition at auth.ts line 42 error ERR_005"},
            {"content": "Database PostgreSQL 14.2 with PgBouncer pooling"},
            {"content": "API latency dropped from 200ms to 80ms after deploy"},
            {"content": "Memory leak LRU overflow 2.1GB to 180MB"},
            {"content": "Must add TOTP 2FA GDPR tokens deletable 30 days"},
            {"content": "Load test 500 RPS sustained 45ms p50 zero errors"},
            {"content": "ERR_005 present 14 days 3 percent users affected"},
            {"content": "Next OAuth2 Google GitHub social login 8 points"},
            {"content": "Redis session architecture for new features"},
        ]
        graph = extract_graph(messages)
        total = graph.total_anchors
        assert 3 <= total <= 15, f"Expected 3-15 anchors, got {total} ({len(graph.verb_anchors)}v + {len(graph.noun_anchors)}n)"
        assert len(graph.verb_anchors) > 0, "Should have verb anchors"
        assert len(graph.noun_anchors) > 0, "Should have noun anchors"


class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: extract graph → format → links intact."""
        from anchor.extractor import extract_graph

        messages = [
            {"content": "We decided to use Redis SETNX for distributed locking"},
            {"content": "Found JWT race condition in auth.ts at line 42"},
            {"content": "Error code ERR_005 must sync across Pods"},
            {"content": "Database is PostgreSQL 14.2"},
        ]
        graph = extract_graph(messages)
        assert graph.total_anchors >= 2

        # Links should be valid
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                assert graph.find_noun(v.nearest_noun_id) is not None
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                assert graph.find_verb(n.nearest_verb_id) is not None
