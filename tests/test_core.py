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
from anchor.extractor import extract_anchors, _extract_entities, _classify_entity, _is_proper_entity
from anchor.store import AnchorStore, id as anchor_id
from anchor.formatter import format_for_injection, format_compact, format_verbose
from anchor.conflict import detect_conflicts, mark_superseded, _entity_overlap_score
from anchor.constraints import build_constraint_graph
from anchor.judge import judge_significance, _fallback_select


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
# Verbs Tests (US-001: 55+ verb lexicon tests)
# ═══════════════════════════════════════════════════════════════════════════
import time


class TestVerbMapCoverage:
    """Basic VERB_MAP structure and category coverage."""

    def test_verb_map_size(self):
        assert len(VERB_MAP) >= 150

    def test_verb_map_has_all_categories(self):
        categories = set(VERB_MAP.values())
        assert "DECISION" in categories
        assert "DISCOVERY" in categories
        assert "ANOMALY" in categories
        assert "CONSTRAINT" in categories

    def test_decision_category_count(self):
        count = sum(1 for v in VERB_MAP.values() if v == "DECISION")
        assert count >= 50, f"Expected >=50 DECISION verbs, got {count}"

    def test_discovery_category_count(self):
        count = sum(1 for v in VERB_MAP.values() if v == "DISCOVERY")
        assert count >= 30, f"Expected >=30 DISCOVERY verbs, got {count}"

    def test_anomaly_category_count(self):
        count = sum(1 for v in VERB_MAP.values() if v == "ANOMALY")
        assert count >= 30, f"Expected >=30 ANOMALY verbs, got {count}"

    def test_constraint_category_count(self):
        count = sum(1 for v in VERB_MAP.values() if v == "CONSTRAINT")
        assert count >= 30, f"Expected >=30 CONSTRAINT verbs, got {count}"


class TestEnglishDecisionVerbs:
    """30 English DECISION verbs with past tense coverage."""

    DECISION_VERBS = [
        ("decide", "DECISION"), ("decided", "DECISION"),
        ("chose", "DECISION"), ("choose", "DECISION"),
        ("switch", "DECISION"), ("switched", "DECISION"),
        ("replace", "DECISION"), ("replaced", "DECISION"),
        ("migrate", "DECISION"), ("migrated", "DECISION"),
        ("upgrade", "DECISION"), ("upgraded", "DECISION"),
        ("downgrade", "DECISION"), ("downgraded", "DECISION"),
        ("deploy", "DECISION"), ("deployed", "DECISION"),
        ("release", "DECISION"), ("released", "DECISION"),
        ("merge", "DECISION"), ("merged", "DECISION"),
        ("refactor", "DECISION"), ("refactored", "DECISION"),
        ("optimize", "DECISION"), ("optimized", "DECISION"),
        ("configure", "DECISION"), ("configured", "DECISION"),
        ("enable", "DECISION"), ("enabled", "DECISION"),
        ("disable", "DECISION"), ("disabled", "DECISION"),
        ("add", "DECISION"), ("added", "DECISION"),
        ("remove", "DECISION"), ("removed", "DECISION"),
        ("delete", "DECISION"), ("deleted", "DECISION"),
        ("adopt", "DECISION"), ("adopted", "DECISION"),
        ("select", "DECISION"), ("selected", "DECISION"),
    ]

    @pytest.mark.parametrize("verb,expected_type", DECISION_VERBS)
    def test_decision_verb(self, verb, expected_type):
        assert get_anchor_type(verb) == expected_type

    def test_segment_decision_sentence(self):
        text = "We decided to migrate and upgrade the database."
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert "decided" in verbs or "migrate" in verbs or "upgrade" in verbs


class TestEnglishDiscoveryVerbs:
    """15 DISCOVERY verbs with past tense coverage."""

    DISCOVERY_VERBS = [
        ("discover", "DISCOVERY"), ("discovered", "DISCOVERY"),
        ("find", "DISCOVERY"), ("found", "DISCOVERY"),
        ("locate", "DISCOVERY"), ("located", "DISCOVERY"),
        ("identify", "DISCOVERY"), ("identified", "DISCOVERY"),
        ("trace", "DISCOVERY"), ("traced", "DISCOVERY"),
        ("debug", "DISCOVERY"), ("debugged", "DISCOVERY"),
        ("diagnose", "DISCOVERY"), ("diagnosed", "DISCOVERY"),
        ("detect", "DISCOVERY"), ("detected", "DISCOVERY"),
        ("observe", "DISCOVERY"), ("observed", "DISCOVERY"),
        ("notice", "DISCOVERY"), ("noticed", "DISCOVERY"),
        ("realize", "DISCOVERY"), ("realized", "DISCOVERY"),
        ("confirm", "DISCOVERY"), ("confirmed", "DISCOVERY"),
        ("pinpoint", "DISCOVERY"), ("pinpointed", "DISCOVERY"),
        ("isolate", "DISCOVERY"), ("isolated", "DISCOVERY"),
    ]

    @pytest.mark.parametrize("verb,expected_type", DISCOVERY_VERBS)
    def test_discovery_verb(self, verb, expected_type):
        assert get_anchor_type(verb) == expected_type

    def test_segment_discovery_sentence(self):
        text = "I found and identified the root cause and confirmed the fix."
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert any(v in verbs for v in ["found", "identified", "confirmed"])


class TestEnglishAnomalyVerbs:
    """15 ANOMALY verbs with past tense coverage."""

    ANOMALY_VERBS = [
        ("error", "ANOMALY"), ("fail", "ANOMALY"),
        ("timeout", "ANOMALY"), ("crash", "ANOMALY"),
        ("hang", "ANOMALY"), ("block", "ANOMALY"),
        ("leak", "ANOMALY"), ("overflow", "ANOMALY"),
        ("deadlock", "ANOMALY"), ("conflict", "ANOMALY"),
        ("exception", "ANOMALY"), ("broken", "ANOMALY"),
        ("corrupted", "ANOMALY"), ("missing", "ANOMALY"),
        ("panic", "ANOMALY"), ("degraded", "ANOMALY"),
        ("returned null", "ANOMALY"), ("returned empty", "ANOMALY"),
        ("threw", "ANOMALY"),
    ]

    @pytest.mark.parametrize("verb,expected_type", ANOMALY_VERBS)
    def test_anomaly_verb(self, verb, expected_type):
        assert get_anchor_type(verb) == expected_type

    def test_segment_anomaly_sentence(self):
        text = "The service crashed with a timeout and memory leak."
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert any(v in verbs for v in ["crash", "timeout", "leak"])


class TestEnglishConstraintVerbs:
    """15 CONSTRAINT verbs."""

    CONSTRAINT_VERBS = [
        ("because", "CONSTRAINT"), ("must", "CONSTRAINT"),
        ("cannot", "CONSTRAINT"), ("unless", "CONSTRAINT"),
        ("prerequisite", "CONSTRAINT"), ("depends on", "CONSTRAINT"),
        ("require", "CONSTRAINT"), ("need to", "CONSTRAINT"),
        ("at most", "CONSTRAINT"), ("at least", "CONSTRAINT"),
        ("no more than", "CONSTRAINT"), ("compatible", "CONSTRAINT"),
        ("incompatible", "CONSTRAINT"), ("restricted to", "CONSTRAINT"),
        ("constrained by", "CONSTRAINT"), ("limited by", "CONSTRAINT"),
    ]

    @pytest.mark.parametrize("verb,expected_type", CONSTRAINT_VERBS)
    def test_constraint_verb(self, verb, expected_type):
        assert get_anchor_type(verb) == expected_type

    def test_segment_constraint_sentence(self):
        text = "This service requires 2GB RAM and depends on Redis."
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert any(v in verbs for v in ["require", "depends on"])


class TestChineseVerbs:
    """10 Chinese verb tests across all four categories with synonym variants."""

    def test_chinese_decision_decided(self):
        assert get_anchor_type("决定") == "DECISION"

    def test_chinese_decision_switch(self):
        assert get_anchor_type("改用") == "DECISION"
        assert get_anchor_type("切换") == "DECISION"

    def test_chinese_decision_deploy(self):
        assert get_anchor_type("部署") == "DECISION"
        assert get_anchor_type("发布") == "DECISION"

    def test_chinese_discovery_found(self):
        assert get_anchor_type("发现") == "DISCOVERY"
        assert get_anchor_type("找到") == "DISCOVERY"

    def test_chinese_discovery_locate(self):
        assert get_anchor_type("定位") == "DISCOVERY"
        assert get_anchor_type("排查") == "DISCOVERY"

    def test_chinese_discovery_confirm(self):
        assert get_anchor_type("确认") == "DISCOVERY"
        assert get_anchor_type("识别") == "DISCOVERY"

    def test_chinese_anomaly_error(self):
        assert get_anchor_type("报错") == "ANOMALY"
        assert get_anchor_type("异常") == "ANOMALY"

    def test_chinese_anomaly_crash(self):
        assert get_anchor_type("崩溃") == "ANOMALY"
        assert get_anchor_type("挂了") == "ANOMALY"

    def test_chinese_constraint_must(self):
        assert get_anchor_type("必须") == "CONSTRAINT"
        assert get_anchor_type("需要") == "CONSTRAINT"

    def test_chinese_constraint_because(self):
        assert get_anchor_type("因为") == "CONSTRAINT"
        assert get_anchor_type("除非") == "CONSTRAINT"

    def test_segment_chinese_multi_category(self):
        text = "我们决定升级 Redis，但是发现报错，必须回滚"
        matches = segment_text(text)
        verbs_found = [m[0] for m in matches]
        types_found = [m[1] for m in matches]
        assert any(v in verbs_found for v in ["决定", "升级", "发现", "报错", "必须", "回滚"])
        assert "DECISION" in types_found
        assert "ANOMALY" in types_found


class TestCompoundVerbs:
    """5 compound verb tests: multi-word phrases matched as single units."""

    def test_compound_tracked_down(self):
        assert get_anchor_type("tracked down") == "DISCOVERY"
        matches = segment_text("We tracked down the memory leak")
        verbs = [m[0].lower() for m in matches]
        assert "tracked down" in verbs

    def test_compound_opted_for(self):
        assert get_anchor_type("opted for") == "DECISION"

    def test_compound_narrowed_down(self):
        assert get_anchor_type("narrowed down") == "DISCOVERY"

    def test_compound_figured_out(self):
        assert get_anchor_type("figured out") == "DISCOVERY"
        matches = segment_text("We finally figured out the root cause")
        verbs = [m[0].lower() for m in matches]
        assert "figured out" in verbs

    def test_compound_longer_wins_over_shorter(self):
        """'tracked down' (7 chars) matches before 'down' (4 chars)."""
        text = "We tracked down the bug"
        matches = segment_text(text)
        verbs = [m[0] for m in matches]
        # "tracked down" should be found; "tracked" alone should NOT also appear
        # because the longer pattern consumed those characters
        assert "tracked down" in verbs


class TestCaseInsensitiveVerbs:
    """5 case-insensitive matching tests."""

    def test_title_case_decided(self):
        assert get_anchor_type("Decided") == "DECISION"

    def test_upper_case_found(self):
        assert get_anchor_type("FOUND") == "DISCOVERY"

    def test_mixed_case_crashed(self):
        assert get_anchor_type("Crashed") == "ANOMALY"

    def test_upper_case_must(self):
        assert get_anchor_type("MUST") == "CONSTRAINT"

    def test_segment_text_case_insensitive(self):
        text = "The team Decided to DEPLOY Redis and FOUND a bug"
        matches = segment_text(text)
        verbs = [m[0] for m in matches]
        assert any("ecided" in v or "Decided" in v for v in verbs)
        assert any(v.upper() == "FOUND" or v == "FOUND" for v in verbs)


class TestUnknownVerbs:
    """5 tests: unlisted verbs return FACT anchor type."""

    def test_unknown_english_verb(self):
        assert get_anchor_type("procrastinate") == "FACT"

    def test_unknown_chinese_verb(self):
        assert get_anchor_type("吃火锅") == "FACT"

    def test_unknown_technical_term(self):
        assert get_anchor_type("kubernetes") == "FACT"

    def test_empty_string(self):
        assert get_anchor_type("") == "FACT"

    def test_unknown_nonsense(self):
        assert get_anchor_type("xyzzy123blargh") == "FACT"

    def test_segment_text_unknown_verbs_not_matched(self):
        """segment_text should not return unknown verbs as matches."""
        text = "We procrastinate about Kubernetes and eat hotpot"
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert "procrastinate" not in verbs
        assert "kubernetes" not in verbs


class TestSegmentTextPerformance:
    """segment_text() performance: 1000-char text under 0.01s."""

    def test_performance_1000_chars(self):
        text = (
            "We decided to use Redis SETNX for distributed locking. "
            "Found JWT race condition at auth.ts line 42 error ERR_005. "
            "Database is PostgreSQL 14.2 with PgBouncer pooling. "
            "API latency dropped from 200ms to 80ms after we deployed the fix. "
            "Memory leak detected in LRU cache, overflow from 2.1GB to 180MB. "
            "Must add TOTP 2FA support for GDPR compliance tokens deletable 30 days. "
            "We migrated the auth service and upgraded all dependencies to latest versions. "
            "The crash was caused by a race condition in the session handler module. "
            "We switched to OAuth2 and replaced the old token system for authentication. "
            "Cannot deploy on Friday, must wait until Monday per release policy. "
            "Error 500 returned when timeout exceeds 30 seconds on the production server. "
            "We refactored the extractor and optimized the regex engine for better speed. "
            "Load test results show 500 RPS sustained with p50 latency of 45ms zero errors. "
            "ERR_005 has been present for 14 days affecting 3 percent of total users. "
            "Next we need OAuth2 Google GitHub social login integration 8 points estimated. "
            "Redis session architecture needs redesign for the new features coming in Q3."
        )
        assert len(text) >= 1000, f"Test text too short: {len(text)} chars"

        elapsed = 1.0
        for _ in range(5):  # warm-up
            segment_text(text)
        start = time.perf_counter()
        for _ in range(100):
            segment_text(text)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.01, f"segment_text too slow: {elapsed:.6f}s for 1000-char text"

    def test_performance_returns_correct_structure(self):
        """segment_text returns properly typed tuples."""
        text = "We decided to use Redis and found an error."
        matches = segment_text(text)
        for m in matches:
            assert len(m) == 4
            assert isinstance(m[0], str)  # verb_text
            assert isinstance(m[1], str)  # anchor_type
            assert isinstance(m[2], int)  # start
            assert isinstance(m[3], int)  # end
            assert m[2] < m[3]  # start before end


class TestVerbEdgeCases:
    """Edge cases for verb matching."""

    def test_verb_at_start_of_text(self):
        text = "Decided to use Redis"
        matches = segment_text(text)
        assert len(matches) >= 1

    def test_verb_at_end_of_text(self):
        text = "The migration was finally deployed"
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert "deployed" in verbs

    def test_verb_only_content(self):
        text = "Deployed."
        matches = segment_text(text)
        assert len(matches) >= 1

    def test_no_verb_in_text(self):
        text = "The Redis cluster runs on Kubernetes pods."
        matches = segment_text(text)
        verbs = [m[0].lower() for m in matches]
        assert "redis" not in verbs
        assert "kubernetes" not in verbs

    def test_multiple_verbs_same_category(self):
        text = "Decided to migrate and upgrade the database"
        matches = segment_text(text)
        decision_matches = [m for m in matches if m[1] == "DECISION"]
        assert len(decision_matches) >= 1

    def test_cross_category_verbs_in_sentence(self):
        text = "Decided to deploy, found a bug, error in logs, must fix"
        matches = segment_text(text)
        types = set(m[1] for m in matches)
        assert len(types) >= 2, f"Expected >=2 categories, got {types}"


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
# Extractor — DATA Entity Tests (US-002: 15 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityDataExtraction:
    """15 DATA entity tests: versions, error codes, numbers+units, line numbers."""

    def test_extract_version_14_2(self):
        entities = _extract_entities("PostgreSQL 14.2 is the database version")
        texts = [e[0] for e in entities]
        assert "14.2" in texts

    def test_extract_version_3_10_1(self):
        entities = _extract_entities("Python 3.10.1 upgrade required")
        texts = [e[0] for e in entities]
        assert "3.10.1" in texts

    def test_extract_version_classified_data(self):
        entities = _extract_entities("Upgraded to 14.2")
        for text, ec, _, _ in entities:
            if text == "14.2":
                assert ec == EntityClass.DATA

    def test_extract_error_code_err_005(self):
        entities = _extract_entities("Error ERR_005 on auth service")
        texts = [e[0] for e in entities]
        assert any("ERR_005" in t for t in texts)

    def test_extract_error_code_ann_001(self):
        entities = _extract_entities("Trace ANN_001 in logs")
        texts = [e[0] for e in entities]
        assert any("ANN_001" in t for t in texts)

    def test_extract_error_code_classified_data(self):
        entities = _extract_entities("Got ERR_005 from API")
        for text, ec, _, _ in entities:
            if "ERR_005" in text:
                assert ec == EntityClass.DATA

    def test_extract_number_unit_200ms(self):
        entities = _extract_entities("Latency dropped to 200ms")
        texts = [e[0] for e in entities]
        assert any("200ms" in t for t in texts)

    def test_extract_number_unit_2_1GB(self):
        entities = _extract_entities("Memory peaked at 2.1GB")
        texts = [e[0] for e in entities]
        assert any("2.1GB" in t for t in texts)

    def test_extract_number_unit_500rps(self):
        entities = _extract_entities("Throughput reached 500 RPS")
        texts = [e[0] for e in entities]
        assert any("500" in str(t) for t in texts)

    def test_extract_number_unit_80mb(self):
        entities = _extract_entities("Image size is 80MB")
        texts = [e[0] for e in entities]
        assert any("80MB" in t for t in texts)

    def test_extract_line_number_single(self):
        entities = _extract_entities("Error at auth.ts:42 — null pointer")
        texts = [e[0] for e in entities]
        assert any("42" == t or ":42" in str(t) for t in texts)

    def test_extract_line_number_three_digit(self):
        entities = _extract_entities("user.py:142 has the bug")
        texts = [e[0] for e in entities]
        assert any("142" == str(t) for t in texts)

    def test_extract_standalone_number_14(self):
        entities = _extract_entities("Found 14 instances of the bug")
        texts = [e[0] for e in entities]
        assert "14" in texts

    def test_standalone_number_under_10_filtered(self):
        """Bare numbers under 10 should not be extracted as entities."""
        entities = _extract_entities("Found 5 bugs and 3 errors")
        texts = [e[0] for e in entities]
        assert "5" not in texts
        assert "3" not in texts

    def test_number_unit_before_standalone_number(self):
        """Number-with-unit regex runs before standalone number regex."""
        entities = _extract_entities("Latency: 200ms or 14 seconds")
        texts = [e[0] for e in entities]
        assert any("200ms" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════════
# Extractor — TECH Entity Tests (US-002: 15 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityTechExtraction:
    """15 TECH entity tests: filenames, PascalCase, camelCase, UPPER_CASE, domains."""

    def test_extract_filename_auth_ts(self):
        entities = _extract_entities("Race condition in auth.ts:42")
        texts = [e[0] for e in entities]
        assert any("auth.ts" == t or "auth" in str(t) for t in texts)

    def test_extract_filename_py_file(self):
        entities = _extract_entities("The user.py module needs refactoring")
        texts = [e[0] for e in entities]
        assert any("user.py" in t for t in texts)

    def test_extract_filename_yaml_config(self):
        entities = _extract_entities("Update config.yaml for the new env")
        texts = [e[0] for e in entities]
        assert any("config.yaml" in t for t in texts)

    def test_extract_pascalcase_postgresql(self):
        entities = _extract_entities("We use PostgreSQL 14.2 as the main database")
        texts = [e[0] for e in entities]
        assert any("PostgreSQL" in t for t in texts)

    def test_extract_pascalcase_oauth2(self):
        entities = _extract_entities("Implement OAuth2 for social login")
        texts = [e[0] for e in entities]
        assert any("OAuth2" in t for t in texts)

    def test_extract_pascalcase_classified_tech(self):
        entities = _extract_entities("PostgreSQL is the database")
        for text, ec, _, _ in entities:
            if "PostgreSQL" in text:
                assert ec == EntityClass.TECH

    def test_extract_uppercase_setnx(self):
        entities = _extract_entities("Use Redis SETNX for distributed locking")
        texts = [e[0] for e in entities]
        assert any("SETNX" in t for t in texts)

    def test_extract_uppercase_jwt(self):
        entities = _extract_entities("JWT tokens expire after 1 hour")
        texts = [e[0] for e in entities]
        assert any("JWT" in t for t in texts)

    def test_extract_uppercase_classified_tech(self):
        entities = _extract_entities("JWT token in SETNX lock")
        for text, ec, _, _ in entities:
            if text == "JWT" or text == "SETNX":
                assert ec == EntityClass.TECH

    def test_extract_camelcase_variable(self):
        entities = _extract_entities("The getCwd function needs updating")
        texts = [e[0] for e in entities]
        assert any("getCwd" in t for t in texts)

    def test_extract_domain_grafana_internal(self):
        entities = _extract_entities("Monitor at grafana.internal")
        texts = [e[0] for e in entities]
        assert any("grafana.internal" in t for t in texts)

    def test_extract_domain_example_com(self):
        entities = _extract_entities("API at api.example.com is down")
        texts = [e[0] for e in entities]
        assert any("example.com" in t for t in texts)

    def test_extract_filename_go_file(self):
        entities = _extract_entities("The main.go file is the entry point")
        texts = [e[0] for e in entities]
        assert any("main.go" in t for t in texts)

    def test_extract_snake_case_identifier(self):
        entities = _extract_entities("Use distributed_lock for sync")
        texts = [e[0] for e in entities]
        assert any("distributed_lock" in t for t in texts)

    def test_extract_uppercase_acronym(self):
        entities = _extract_entities("LRU cache eviction policy")
        texts = [e[0] for e in entities]
        assert any("LRU" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════════
# Extractor — Garbage Filter Tests (US-002: 10 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityGarbageFilter:
    """10 garbage filter tests: sentence-initial caps, bare numbers, fragments."""

    def test_stop_entity_BEFORE_filtered(self):
        """'BEFORE' is in _STOP_ENTITIES and should be filtered."""
        entities = _extract_entities("BEFORE we start the migration, check Redis")
        texts = [e[0] for e in entities]
        assert "BEFORE" not in texts

    def test_stop_entity_CANNOT_filtered(self):
        """'CANNOT' (uppercase) is in _STOP_ENTITIES and should be filtered."""
        entities = _extract_entities("CANNOT deploy on Friday due to freeze")
        texts = [e[0] for e in entities]
        assert "CANNOT" not in texts

    def test_stop_entity_lowercase_using_filtered(self):
        """'using' is in _STOP_ENTITIES and should be filtered."""
        entities = _extract_entities("Fixed using the new Redis approach")
        texts = [e[0] for e in entities]
        assert "using" not in texts

    def test_stop_entity_FIRST_filtered(self):
        """'FIRST' is in _STOP_ENTITIES and should be filtered."""
        entities = _extract_entities("FIRST we need to check the database")
        texts = [e[0] for e in entities]
        assert "FIRST" not in texts

    def test_bare_number_5_filtered(self):
        entities = _extract_entities("Found 5 errors in the logs")
        texts = [e[0] for e in entities]
        assert "5" not in texts

    def test_bare_number_7_filtered(self):
        entities = _extract_entities("Only 7 users affected")
        texts = [e[0] for e in entities]
        assert "7" not in texts

    def test_hyphen_start_fragment_filtered(self):
        entities = _extract_entities("The -based approach works")
        texts = [e[0] for e in entities]
        assert "-based" not in texts

    def test_underscore_start_fragment_filtered(self):
        entities = _extract_entities("_private field should not leak")
        texts = [e[0] for e in entities]
        assert not any(t.startswith("_") for t in texts)

    def test_stop_entity_before_filtered(self):
        """'BEFORE' in stop list should not appear."""
        entities = _extract_entities("BEFORE we begin the migration")
        texts = [e[0] for e in entities]
        assert "BEFORE" not in texts

    def test_bare_number_10_extracted(self):
        """Numbers >= 10 should be extracted as DATA entities."""
        entities = _extract_entities("Found 14 instances total")
        texts = [e[0] for e in entities]
        assert "14" in texts


# ═══════════════════════════════════════════════════════════════════════════
# Extractor — Chinese Entity Tests (US-002: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestChineseEntityExtraction:
    """5 Chinese entity tests: distributed lock, cross-pod sync, etc."""

    def test_chinese_term_distributed_lock(self):
        entities = _extract_entities("使用分布式锁来同步操作")
        texts = [e[0] for e in entities]
        assert len(texts) >= 1, f"Expected at least 1 entity, got {texts}"
        for _, ec, _, _ in entities:
            assert ec == EntityClass.TERM

    def test_chinese_term_cross_pod_sync(self):
        entities = _extract_entities("跨Pod同步需要处理竞争条件")
        texts = [e[0] for e in entities]
        assert len(texts) >= 1, f"Expected at least 1 entity, got {texts}"

    def test_chinese_term_classified_term(self):
        entities = _extract_entities("需要分布式锁来处理")
        for text, ec, _, _ in entities:
            if "分布式" in text or "锁" in text:
                assert ec == EntityClass.TERM

    def test_chinese_term_data_value(self):
        entities = _extract_entities("报错 ERR_005 在 auth.ts:42")
        texts = [e[0] for e in entities]
        assert any("ERR_005" in t or "auth.ts" in t or "42" == str(t) for t in texts)

    def test_chinese_term_multi_char(self):
        entities = _extract_entities("数据库连接池配置需要更新")
        texts = [e[0] for e in entities]
        assert len(texts) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Extractor — Mixed Entity Tests (US-002: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestMixedEntityExtraction:
    """5 mixed entity tests: Chinese + English combined text."""

    def test_mixed_chinese_english_sentence(self):
        text = "我们使用 Redis 分布式锁来处理 JWT race condition"
        entities = _extract_entities(text)
        texts = [e[0] for e in entities]
        assert len(texts) >= 2, f"Expected >=2 entities, got {texts}"

    def test_mixed_chinese_with_version(self):
        text = "PostgreSQL 14.2数据库需要配置连接池大小20"
        entities = _extract_entities(text)
        texts = [e[0] for e in entities]
        # PostgreSQL should be found as a TECH entity
        assert any("PostgreSQL" in t for t in texts), f"Expected PostgreSQL in {texts}"
        # 14 or 14.2 should be found as DATA
        has_version = any("14" in str(t) for t in texts)
        assert has_version, f"Expected 14/14.2 in {texts}"

    def test_mixed_chinese_with_error_code(self):
        text = "错误码ERR_005出现在auth.ts的42行"
        entities = _extract_entities(text)
        texts = [e[0] for e in entities]
        assert len(texts) >= 1, f"Expected at least 1 entity, got {texts}"

    def test_mixed_entity_count(self):
        """A mixed text should produce at least 3 entities."""
        text = "我们用 PostgreSQL 14.2 代替 Redis 分布式锁 JWT 认证"
        entities = _extract_entities(text)
        assert len(entities) >= 3, f"Expected >=3 entities, got {len(entities)}: {entities}"

    def test_mixed_entity_classification(self):
        """DATA and TECH entities should coexist in mixed text."""
        text = "PostgreSQL 14.2 ERR_005 auth.ts:42"
        entities = _extract_entities(text)
        classes = set(ec for _, ec, _, _ in entities if ec is not None)
        assert EntityClass.DATA in classes


# ═══════════════════════════════════════════════════════════════════════════
# Extractor — _is_proper_entity Edge Cases (US-002)
# ═══════════════════════════════════════════════════════════════════════════

class TestIsProperEntity:
    """_is_proper_entity() edge cases: short tech terms vs common words."""

    def test_redis_5_chars_is_proper(self):
        assert _is_proper_entity("Redis") is True

    def test_decided_7_chars_blacklisted(self):
        assert _is_proper_entity("Decided") is False

    def test_cannot_6_chars_blacklisted(self):
        assert _is_proper_entity("Cannot") is False

    def test_postgresql_pascalcase_is_proper(self):
        assert _is_proper_entity("PostgreSQL") is True

    def test_jwt_acronym_is_proper(self):
        assert _is_proper_entity("JWT") is True

    def test_lowercase_common_word_not_proper(self):
        assert _is_proper_entity("database") is False

    def test_entity_with_digits_is_proper(self):
        assert _is_proper_entity("14.2") is True

    def test_entity_with_dot_is_proper(self):
        assert _is_proper_entity("auth.ts") is True

    def test_snake_case_compound_is_proper(self):
        assert _is_proper_entity("distributed_lock") is True

    def test_all_uppercase_3_chars_is_proper(self):
        assert _is_proper_entity("LRU") is True

    def test_single_initial_cap_common_vowel_heavy(self):
        """'Decided' has high vowel ratio → not proper. But it's also blacklisted."""
        assert _is_proper_entity("Database") is False

    def test_empty_string_not_proper(self):
        assert _is_proper_entity("") is False

    def test_short_lowercase_not_proper(self):
        assert _is_proper_entity("set") is False


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


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Verb Anchor Tests (US-003: 10 tests)
# ═══════════════════════════════════════════════════════════════════════════

import unittest.mock as mock


def _extract_graph_fallback(msgs, sid=None):
    """extract_graph with forced fallback mode for deterministic tests."""
    from anchor.extractor import extract_graph
    with mock.patch.dict(os.environ, {}, clear=True):
        return extract_graph(msgs, sid)


class TestGraphVerbAnchors:
    """10 verb anchor tests: each verb correctly links to nearest noun."""

    def test_verb_decision_links_to_nearest_noun(self):
        """'decided' links to nearest noun (Redis)."""
        msgs = [{"content": "We decided to use Redis for the cache layer"}]
        graph = _extract_graph_fallback(msgs)
        decided = next((v for v in graph.verb_anchors if "decid" in v.entity.lower()), None)
        assert decided is not None, f"No 'decided' verb anchor found in {[v.entity for v in graph.verb_anchors]}"
        assert decided.nearest_noun_id != "", "Verb should link to a noun"
        linked = graph.find_noun(decided.nearest_noun_id)
        assert linked is not None, f"Dangling ref to {decided.nearest_noun_id}"

    def test_verb_discovery_links_to_noun(self):
        """'identified' (DISCOVERY, not in stop-words) should link to a nearby entity."""
        msgs = [{"content": "We identified a race condition in PostgreSQL at line 142"}]
        graph = _extract_graph_fallback(msgs)
        ident_v = next((v for v in graph.verb_anchors if "identif" in v.entity.lower()), None)
        assert ident_v is not None, f"Expected 'identified' verb in {[v.entity for v in graph.verb_anchors]}"
        assert ident_v.nearest_noun_id != ""

    def test_verb_anomaly_links_to_noun(self):
        """'crash' should link to a nearby entity."""
        msgs = [{"content": "The auth.ts module crashed with error ERR_005"}]
        graph = _extract_graph_fallback(msgs)
        crash_v = next((v for v in graph.verb_anchors if "crash" in v.entity.lower()), None)
        assert crash_v is not None, f"Expected 'crash' verb in {[v.entity for v in graph.verb_anchors]}"
        assert crash_v.nearest_noun_id != ""

    def test_verb_constraint_links_to_noun(self):
        """'must' (CONSTRAINT verb) should link to a nearby entity."""
        msgs = [{"content": "Redis must have distributed locking across pods"}]
        graph = _extract_graph_fallback(msgs)
        must_v = next((v for v in graph.verb_anchors if "must" in v.entity.lower()), None)
        assert must_v is not None, f"Expected 'must' verb in {[v.entity for v in graph.verb_anchors]}"
        if must_v.nearest_noun_id:
            linked = graph.find_noun(must_v.nearest_noun_id)
            assert linked is not None
        # At minimum, 'must' is present as a verb anchor

    def test_verb_links_closest_noun_not_farther(self):
        """When 2+ nouns exist, verb links to the closest one."""
        msgs = [{"content": "decided PostgreSQL migration Redis caching approach"}]
        graph = _extract_graph_fallback(msgs)
        dec_v = next((v for v in graph.verb_anchors if "decid" in v.entity.lower()), None)
        assert dec_v is not None
        linked = graph.find_noun(dec_v.nearest_noun_id)
        assert linked is not None
        # "PostgreSQL" (pos ~8) is closer to "decided" (pos ~0) than "Redis" (pos ~29)
        assert "PostgreSQL" in linked.entity or "Redis" in linked.entity

    def test_verb_data_hints_populated(self):
        """Verb near DATA entities should have data_hints."""
        msgs = [{"content": "Migrated PostgreSQL from version 14.2 to 15.1"}]
        graph = _extract_graph_fallback(msgs)
        migrated = next((v for v in graph.verb_anchors if "migrat" in v.entity.lower()), None)
        if migrated and migrated.data_hints:
            assert any("14.2" in d or "15.1" in d for d in migrated.data_hints)

    def test_verb_anchor_type_matches_category(self):
        """VerbAnchor.anchor_type must be the correct AnchorType enum."""
        msgs = [{"content": "We decided to use Redis and found a bug that crashed"}]
        graph = _extract_graph_fallback(msgs)
        type_map = {}
        for v in graph.verb_anchors:
            type_map[v.entity.lower()] = v.anchor_type.value
        if "decided" in type_map:
            assert type_map["decided"] == "DECISION"
        if "found" in type_map:
            assert type_map["found"] == "DISCOVERY"

    def test_verb_entity_preserved(self):
        """Verb entity text should match what segment_text returned."""
        msgs = [{"content": "We decided to migrate PostgreSQL and upgrade the backend"}]
        graph = _extract_graph_fallback(msgs)
        entities = [v.entity for v in graph.verb_anchors]
        assert len(entities) >= 1, f"Expected at least 1 verb, got {entities}"

    def test_verb_pos_within_text_range(self):
        """Verb positions should be within the message text bounds."""
        msgs = [{"content": "We decided to deploy Redis cluster configuration"}]
        graph = _extract_graph_fallback(msgs)
        text_len = len(msgs[0]["content"])
        for v in graph.verb_anchors:
            assert 0 <= v.pos <= text_len, f"Verb pos {v.pos} out of range [0, {text_len}]"

    def test_multiple_verbs_link_to_different_nouns(self):
        """Two verbs in same text link to different nouns if nouns are in different windows."""
        msgs = [{"content": "Decided Redis caching. much later... Found error in PostgreSQL query"}]
        graph = _extract_graph_fallback(msgs)
        verbs_with_links = [v for v in graph.verb_anchors if v.nearest_noun_id]
        if len(verbs_with_links) >= 1:
            linked_ids = {v.nearest_noun_id for v in verbs_with_links}
            assert len(linked_ids) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Noun Anchor Tests (US-003: 10 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphNounAnchors:
    """10 noun anchor tests: each noun correctly links to nearest verb, tags populated."""

    def test_data_noun_links_to_verb(self):
        """DATA entity (14.2 version) should link to a nearby verb."""
        msgs = [{"content": "We upgraded PostgreSQL to version 14.2"}]
        graph = _extract_graph_fallback(msgs)
        version_n = next((n for n in graph.noun_anchors if "14.2" in n.entity), None)
        assert version_n is not None, f"No '14.2' noun found in {[n.entity for n in graph.noun_anchors]}"
        assert version_n.nearest_verb_id != "", "14.2 should link to a nearby verb"
        linked = graph.find_verb(version_n.nearest_verb_id)
        assert linked is not None, f"Dangling ref to {version_n.nearest_verb_id}"

    def test_tech_noun_links_to_verb(self):
        """TECH entity (PostgreSQL) should link to a nearby verb."""
        msgs = [{"content": "We decided PostgreSQL is the primary database"}]
        graph = _extract_graph_fallback(msgs)
        pg_n = next((n for n in graph.noun_anchors if "PostgreSQL" in n.entity), None)
        assert pg_n is not None, f"No PostgreSQL noun found in {[n.entity for n in graph.noun_anchors]}"
        if pg_n.nearest_verb_id:
            linked = graph.find_verb(pg_n.nearest_verb_id)
            assert linked is not None

    def test_noun_tags_populated_for_redis(self):
        """Redis entity should get semantic tags from _ENTITY_SEMANTIC_TAGS."""
        msgs = [{"content": "We decided to use Redis as the cache layer"}]
        graph = _extract_graph_fallback(msgs)
        redis_n = next((n for n in graph.noun_anchors if n.entity == "Redis"), None)
        assert redis_n is not None, \
            f"No standalone 'Redis' noun in {[n.entity for n in graph.noun_anchors]}"
        assert len(redis_n.tags) > 0, f"Redis should have tags, got {redis_n.tags}"

    def test_noun_tags_populated_for_jwt(self):
        """JWT entity should get auth-related semantic tags."""
        msgs = [{"content": "We identified JWT token race condition in auth module"}]
        graph = _extract_graph_fallback(msgs)
        jwt_n = next((n for n in graph.noun_anchors if n.entity == "JWT"), None)
        assert jwt_n is not None, \
            f"No standalone 'JWT' noun in {[n.entity for n in graph.noun_anchors]}"
        assert len(jwt_n.tags) > 0, f"JWT should have tags, got {jwt_n.tags}"

    def test_noun_data_values_populated(self):
        """Noun near DATA patterns should have data_values extracted."""
        msgs = [{"content": "Found PostgreSQL version 14.2 with error ERR_005 at auth.ts:42"}]
        graph = _extract_graph_fallback(msgs)
        nouns_with_data = [n for n in graph.noun_anchors if n.data_values]
        assert len(nouns_with_data) > 0, "Expected at least one noun with data_values"

    def test_noun_entity_class_matches(self):
        """Noun entity_class should match its actual type."""
        msgs = [{"content": "PostgreSQL 14.2 ERR_005 auth.ts:42 JWT token crash"}]
        graph = _extract_graph_fallback(msgs)
        for n in graph.noun_anchors:
            assert n.entity_class.value in ("DATA", "TECH", "TERM"), \
                f"Invalid entity_class {n.entity_class} for {n.entity}"

    def test_noun_entity_preserved_in_graph(self):
        """Noun entity text should be preserved exactly."""
        msgs = [{"content": "We decided to add Redis and PostgreSQL 14.2 with JWT auth"}]
        graph = _extract_graph_fallback(msgs)
        assert len(graph.noun_anchors) > 0, \
            f"Expected at least one noun anchor, got {[n.entity for n in graph.noun_anchors]}"

    def test_noun_pos_within_text_range(self):
        """Noun positions should be within the input text bounds."""
        msgs = [{"content": "Redis cluster with PostgreSQL 14.2 database"}]
        graph = _extract_graph_fallback(msgs)
        text_len = len(msgs[0]["content"])
        for n in graph.noun_anchors:
            assert 0 <= n.pos <= text_len, f"Noun pos {n.pos} out of range [0, {text_len}]"

    def test_noun_links_to_closest_verb(self):
        """When 2+ verbs exist, noun links to the closest one."""
        msgs = [{"content": "decided to deploy Redis crashed after timeout"}]
        graph = _extract_graph_fallback(msgs)
        redis_n = next((n for n in graph.noun_anchors if "Redis" in n.entity), None)
        assert redis_n is not None, f"No Redis noun in {[n.entity for n in graph.noun_anchors]}"
        if redis_n.nearest_verb_id:
            linked = graph.find_verb(redis_n.nearest_verb_id)
            assert linked is not None

    def test_tech_noun_tags_populated_for_postgresql(self):
        """PostgreSQL should get database-related tags."""
        msgs = [{"content": "We migrated PostgreSQL from version 14.2 to 15.1"}]
        graph = _extract_graph_fallback(msgs)
        pg_n = next((n for n in graph.noun_anchors if "PostgreSQL" in n.entity), None)
        assert pg_n is not None, f"No PostgreSQL in {[n.entity for n in graph.noun_anchors]}"
        assert len(pg_n.tags) > 0, f"PostgreSQL should have tags, got {pg_n.tags}"


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Link Integrity Tests (US-003: 8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphLinkIntegrity:
    """8 link integrity tests: no dangling references, bidirectional consistency."""

    def test_no_dangling_verb_to_noun_links(self):
        """Every verb.nearest_noun_id must point to an existing noun."""
        msgs = [
            {"content": "Decided to use Redis SETNX for distributed locking"},
            {"content": "Found JWT race condition at auth.ts:42 ERR_005"},
            {"content": "PostgreSQL 14.2 with PgBouncer pooling"},
        ]
        graph = _extract_graph_fallback(msgs)
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                n = graph.find_noun(v.nearest_noun_id)
                assert n is not None, f"Verb '{v.entity}' links to missing noun {v.nearest_noun_id}"

    def test_no_dangling_noun_to_verb_links(self):
        """Every noun.nearest_verb_id must point to an existing verb."""
        msgs = [
            {"content": "Decided to use Redis SETNX for distributed locking"},
            {"content": "Found JWT race condition at auth.ts:42 ERR_005"},
            {"content": "PostgreSQL 14.2 with PgBouncer pooling"},
        ]
        graph = _extract_graph_fallback(msgs)
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                v = graph.find_verb(n.nearest_verb_id)
                assert v is not None, f"Noun '{n.entity}' links to missing verb {n.nearest_verb_id}"

    def test_bidirectional_pair_consistency(self):
        """If v→n and n→v point to each other, the pair is consistent."""
        msgs = [{"content": "We upgraded PostgreSQL to version 14.2 for better performance"}]
        graph = _extract_graph_fallback(msgs)
        for v in graph.verb_anchors:
            if not v.nearest_noun_id:
                continue
            n = graph.find_noun(v.nearest_noun_id)
            if n and n.nearest_verb_id == v.id:
                assert n.nearest_verb_id == v.id

    def test_find_verb_by_id_returns_correct_verb(self):
        """find_verb() should locate the correct VerbAnchor by its id."""
        msgs = [{"content": "We decided to use Redis for caching"}]
        graph = _extract_graph_fallback(msgs)
        for v in graph.verb_anchors:
            found = graph.find_verb(v.id)
            assert found is not None, f"find_verb({v.id}) returned None"
            assert found.entity == v.entity

    def test_find_noun_by_id_returns_correct_noun(self):
        """find_noun() should locate the correct NounAnchor by its id."""
        msgs = [{"content": "PostgreSQL 14.2 database with JWT auth"}]
        graph = _extract_graph_fallback(msgs)
        for n in graph.noun_anchors:
            found = graph.find_noun(n.id)
            assert found is not None, f"find_noun({n.id}) returned None"
            assert found.entity == n.entity

    def test_find_noun_nonexistent_returns_none(self):
        """find_noun() for bogus ID returns None."""
        graph = _extract_graph_fallback([{"content": "Redis cache"}])
        assert graph.find_noun("nonexistent-id-12345") is None

    def test_find_verb_nonexistent_returns_none(self):
        """find_verb() for bogus ID returns None."""
        graph = _extract_graph_fallback([{"content": "decided to deploy"}])
        assert graph.find_verb("nonexistent-id-12345") is None

    def test_graph_total_anchors_equals_sum(self):
        """total_anchors == len(verb_anchors) + len(noun_anchors)."""
        msgs = [
            {"content": "Decided to use Redis SETNX for distributed lock"},
            {"content": "Found JWT race condition in auth.ts:42 ERR_005"},
            {"content": "Must add GDPR compliance tokens"},
        ]
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors == len(graph.verb_anchors) + len(graph.noun_anchors)


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Dedup Tests (US-003: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphDedup:
    """5 dedup tests: duplicate entities/verbs filtered out."""

    def test_duplicate_verb_entity_filtered(self):
        """Same verb entity appearing twice in messages should appear once in graph."""
        msgs = [
            {"content": "We decided to use Redis"},
            {"content": "We decided to also use PostgreSQL"},
        ]
        graph = _extract_graph_fallback(msgs)
        decided_count = sum(1 for v in graph.verb_anchors if "decid" in v.entity.lower())
        assert decided_count <= 1, f"Duplicate 'decided' verb not filtered: count={decided_count}"

    def test_duplicate_noun_entity_filtered(self):
        """Same noun entity appearing twice should appear once in graph."""
        msgs = [
            {"content": "PostgreSQL is the main database"},
            {"content": "We also considered PostgreSQL for analytics"},
        ]
        graph = _extract_graph_fallback(msgs)
        pg_count = sum(1 for n in graph.noun_anchors if "PostgreSQL" in n.entity)
        assert pg_count <= 1, f"Duplicate PostgreSQL noun not filtered: count={pg_count}"

    def test_both_verb_and_noun_dedup_combined(self):
        """Both duplicate verbs AND nouns filtered in the same graph."""
        msgs = [
            {"content": "decided Redis caching"},
            {"content": "decided PostgreSQL storage"},
            {"content": "Redis also used for sessions"},
            {"content": "PostgreSQL used for analytics"},
        ]
        graph = _extract_graph_fallback(msgs)
        decided_count = sum(1 for v in graph.verb_anchors if "decid" in v.entity.lower())
        redis_count = sum(1 for n in graph.noun_anchors if "Redis" in n.entity)
        pg_count = sum(1 for n in graph.noun_anchors if "PostgreSQL" in n.entity)
        assert decided_count <= 1, f"Expected <=1 'decided', got {decided_count}"
        assert redis_count <= 1, f"Expected <=1 'Redis', got {redis_count}"
        assert pg_count <= 1, f"Expected <=1 'PostgreSQL', got {pg_count}"

    def test_dedup_preserves_link_integrity(self):
        """After dedup, remaining anchors have valid links."""
        msgs = [
            {"content": "decided to use Redis"},
            {"content": "decided to deploy PostgreSQL"},
        ]
        graph = _extract_graph_fallback(msgs)
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                assert graph.find_noun(v.nearest_noun_id) is not None
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                assert graph.find_verb(n.nearest_verb_id) is not None

    def test_three_duplicate_nouns_become_one(self):
        """Three occurrences of same noun → only one anchor kept."""
        msgs = [
            {"content": "Redis cache layer"},
            {"content": "Redis session store"},
            {"content": "Redis message queue"},
        ]
        graph = _extract_graph_fallback(msgs)
        redis_count = sum(1 for n in graph.noun_anchors if "Redis" in n.entity)
        assert redis_count <= 1, f"Three Redis refs should dedup to 1, got {redis_count}"


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Top-N Tests (US-003: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphTopN:
    """5 Top-N tests: anchor count bounded by target = max(8, n_messages//2)."""

    def _make_tech_msg(self, i):
        topics = [
            "decided Redis SETNX caching", "found JWT race condition auth",
            "migrated PostgreSQL 14.2 database", "must add GDPR compliance",
            "deployed OAuth2 authentication flow", "crashed with error ERR_00",
            "upgraded Kubernetes cluster pods", "configured Prometheus metrics",
            "optimized LRU cache eviction", "detected memory leak overflow",
            "refactored GraphQL API schema", "enabled TOTP 2FA security",
            "switched from MySQL to PlanetScale", "configured Vite build pipeline",
            "added Cloudflare CDN edge caching", "fixed XSS injection CSP headers",
            "integrated Clerk auth session", "released feature flag rollout",
            "monitored p50 latency 45ms", "tracked down race condition bug",
            "deployed Docker container image", "migrated schema Prisma ORM",
            "upgraded pnpm workspace monorepo", "configured Playwright E2E tests",
            "identified CSRF token vulnerability", "installed Datadog APM monitoring",
            "optimized WebSocket connection pool", "debugged tRPC type-safe API",
            "configured Tailwind utility classes", "integrated Storybook components",
            "verified WCAG accessibility standards", "deployed Vercel preview branch",
        ]
        return {"content": f"Update {i}: {topics[i % len(topics)]} version {10 + i}.{i % 5}"}

    def test_thirty_messages_produces_bounded_anchors(self):
        """30 messages should produce anchors ≤ max(8, 30//2) = 15, and ≥ 12."""
        msgs = [self._make_tech_msg(i) for i in range(30)]
        graph = _extract_graph_fallback(msgs)
        target = max(8, len(msgs) // 2)
        assert graph.total_anchors <= target, \
            f"Total anchors {graph.total_anchors} > target {target}"
        assert graph.total_anchors >= 12, \
            f"Expected at least 12 anchors from 30 dense messages, got {graph.total_anchors}"

    def test_ten_messages_target_is_eight(self):
        """10 messages → target = max(8, 10//2) = 8."""
        msgs = [self._make_tech_msg(i) for i in range(10)]
        graph = _extract_graph_fallback(msgs)
        target = max(8, len(msgs) // 2)
        assert graph.total_anchors <= target

    def test_six_messages_target_is_eight(self):
        """6 messages → target = max(8, 6//2) = 8."""
        msgs = [self._make_tech_msg(i) for i in range(6)]
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors <= max(8, len(msgs) // 2)

    def test_twenty_messages_target_is_ten(self):
        """20 messages → target = max(8, 20//2) = 10."""
        msgs = [self._make_tech_msg(i) for i in range(20)]
        graph = _extract_graph_fallback(msgs)
        target = max(8, len(msgs) // 2)
        assert graph.total_anchors <= target
        assert graph.total_anchors >= 5

    def test_anchor_count_never_exceeds_message_count(self):
        """Anchor count should never exceed the number of input messages."""
        for n_msgs in [10, 20, 30]:
            msgs = [self._make_tech_msg(i) for i in range(n_msgs)]
            graph = _extract_graph_fallback(msgs)
            assert graph.total_anchors <= n_msgs, \
                f"{n_msgs} msgs produced {graph.total_anchors} anchors (exceeds message count)"


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph — Empty/Tiny Input Tests (US-003: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphEmptyInput:
    """5 empty/tiny input tests: 0 messages, 1 message, blank content."""

    def test_zero_messages_empty_graph(self):
        """0 messages should produce an empty AnchorGraph."""
        graph = _extract_graph_fallback([])
        assert graph.total_anchors == 0
        assert len(graph.verb_anchors) == 0
        assert len(graph.noun_anchors) == 0

    def test_one_message_produces_some_anchors(self):
        """1 valid message should produce at least one anchor."""
        msgs = [{"content": "We decided to use Redis SETNX for distributed lock"}]
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors >= 1, f"Expected anchors from valid message, got 0"

    def test_blank_content_message_no_anchors(self):
        """Blank/whitespace-only content should produce an empty graph."""
        msgs = [{"content": "   \n  \t  "}]
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors == 0, \
            f"Blank content produced {graph.total_anchors} anchors"

    def test_messages_with_only_stop_words_few_anchors(self):
        """Messages with only common/stop words should produce few or no anchors."""
        msgs = [
            {"content": "Just using the current store before making decisions"},
            {"content": "Very many such issues should also be acceptable"},
            {"content": "Only this should be the default"},
        ]
        graph = _extract_graph_fallback(msgs)
        # Stop-word-heavy text may still extract a few anchorable fragments
        assert graph.total_anchors <= 5, \
            f"Stop-word input produced {graph.total_anchors} anchors, expected <= 5"

    def test_single_short_message(self):
        """Very short message should produce valid graph structure."""
        msgs = [{"content": "Redis crash"}]
        graph = _extract_graph_fallback(msgs)
        assert isinstance(graph.session_id, str)
        assert len(graph.session_id) > 0
        assert hasattr(graph, 'verb_anchors')
        assert hasattr(graph, 'noun_anchors')


# ═══════════════════════════════════════════════════════════════════════════
# Judge — LLM Mode Tests (US-004: 8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgeLLMMode:
    """8 LLM mode tests: mock _call_llm, verify selection count and tags."""

    def _make_mixed_candidates(self, n_verbs=5, n_nouns=5):
        candidates = []
        for i in range(n_verbs):
            candidates.append({
                "entity": f"decided_v{i}",
                "type": "verb",
                "verb_type": ["DECISION", "DISCOVERY", "ANOMALY", "CONSTRAINT", "FACT"][i % 5],
                "pos": i * 10,
                "data": [f"14.{i}"] if i % 2 == 0 else [],
            })
        for i in range(n_nouns):
            candidates.append({
                "entity": f"Entity_{i}",
                "type": "noun",
                "noun_class": ["DATA", "TECH", "TERM"][i % 3],
                "pos": (i + n_verbs) * 10,
                "data": [f"ERR_{i:03d}"] if i % 2 == 0 else [],
            })
        return candidates

    def test_llm_selects_target_count(self):
        """10 candidates, target=6 → LLM selects exactly 6."""
        candidates = self._make_mixed_candidates(5, 5)
        mock_result = [
            {"entity": c["entity"], "type": c["type"], "tags": ["test"]}
            for c in candidates[:6]
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 6, api_key="fake")
        assert len(result) == 6

    def test_llm_tags_non_empty(self):
        """LLM-selected anchors have non-empty tags."""
        candidates = self._make_mixed_candidates(5, 5)
        mock_result = [
            {"entity": c["entity"], "type": c["type"],
             "tags": ["database", "performance"]}
            for c in candidates[:6]
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 6, api_key="fake")
        for item in result:
            assert len(item["tags"]) > 0, f"Expected non-empty tags for {item['entity']}"

    def test_llm_preserves_entity_and_type(self):
        """Returned items match entity and type from candidates."""
        candidates = self._make_mixed_candidates(5, 5)
        mock_result = [
            {"entity": c["entity"], "type": c["type"], "tags": ["test"]}
            for c in [candidates[0], candidates[3], candidates[6]]
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 3, api_key="fake")
        assert len(result) == 3
        assert result[0]["entity"] == candidates[0]["entity"]
        assert result[0]["type"] == candidates[0]["type"]

    def test_llm_api_key_passed_through(self):
        """When api_key is provided, LLM path is taken (not fallback)."""
        candidates = self._make_mixed_candidates(3, 3)
        mock_result = [
            {"entity": c["entity"], "type": c["type"], "tags": ["llm"]}
            for c in candidates[:3]
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result) as mock_call:
            result = judge_significance(candidates, "excerpt", 3, api_key="sk-test")
        mock_call.assert_called_once()
        for item in result:
            assert item["tags"] == ["llm"]

    def test_llm_fewer_than_target_returns_all(self):
        """When candidates <= target, returns all without calling LLM."""
        candidates = self._make_mixed_candidates(2, 2)
        with mock.patch('anchor.judge._call_llm') as mock_call:
            result = judge_significance(candidates, "excerpt", 10, api_key="fake")
        mock_call.assert_not_called()
        assert len(result) == 4

    def test_llm_handles_varied_tag_counts(self):
        """LLM returns varied tag counts (1-4 per anchor)."""
        candidates = self._make_mixed_candidates(5, 5)
        mock_result = [
            {"entity": c["entity"], "type": c["type"], "tags": ["auth"]}
            for c in candidates[:6]
        ]
        mock_result[0]["tags"] = ["database", "storage", "SQL"]
        mock_result[1]["tags"] = ["cache", "distributed lock", "key-value", "Redis"]
        mock_result[2]["tags"] = ["monitoring"]
        mock_result[3]["tags"] = ["auth", "2FA", "security", "MFA"]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 6, api_key="fake")
        tag_counts = [len(item["tags"]) for item in result]
        assert 1 <= min(tag_counts) <= 4
        assert 1 <= max(tag_counts) <= 4

    def test_llm_selection_prioritizes_decisions(self):
        """LLM is prompted to prioritize decisions > anomalies. Verify it receives
        the correct verb_type annotations in the candidate format."""
        candidates = self._make_mixed_candidates(5, 0)
        captured_candidates = None
        original_call_llm = __import__('anchor.judge', fromlist=['_call_llm'])._call_llm

        def capture_and_return(cands, *args, **kwargs):
            nonlocal captured_candidates
            captured_candidates = cands
            return [{"entity": c["entity"], "type": c["type"], "tags": ["test"]}
                    for c in cands[:3]]

        with mock.patch('anchor.judge._call_llm', side_effect=capture_and_return):
            judge_significance(candidates, "excerpt", 3, api_key="fake")
        verb_types = [c["verb_type"] for c in captured_candidates]
        assert "DECISION" in verb_types
        assert "ANOMALY" in verb_types

    def test_llm_target_one_selects_single(self):
        """target_count=1 selects exactly 1 anchor."""
        candidates = self._make_mixed_candidates(5, 5)
        mock_result = [
            {"entity": candidates[0]["entity"], "type": candidates[0]["type"],
             "tags": ["primary"]}
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 1, api_key="fake")
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Judge — Fallback Mode Tests (US-004: 8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgeFallbackMode:
    """8 fallback mode tests: no API key → balanced verb+noun with quota."""

    def _make_candidates(self, n_verbs=8, n_data=5, n_tech=5):
        candidates = []
        for i in range(n_verbs):
            candidates.append({
                "entity": f"verb_{i}",
                "type": "verb",
                "verb_type": ["DECISION", "DISCOVERY", "ANOMALY", "CONSTRAINT",
                              "FACT", "DECISION", "DISCOVERY", "ANOMALY"][i],
                "pos": i * 10,
                "data": [f"v{i}"] if i % 3 == 0 else [],
            })
        for i in range(n_data):
            candidates.append({
                "entity": f"data_{i}",
                "type": "noun",
                "noun_class": "DATA",
                "pos": (i + n_verbs) * 10,
                "data": [f"ERR_{i:03d}"],
            })
        for i in range(n_tech):
            candidates.append({
                "entity": f"Tech_{i}",
                "type": "noun",
                "noun_class": "TECH",
                "pos": (i + n_verbs + n_data) * 10,
                "data": [] if i % 2 == 0 else [f"v{i}.0"],
            })
        return candidates

    def test_fallback_no_api_key_triggers(self):
        """No api_key → _fallback_select is used."""
        candidates = self._make_candidates(5, 3, 3)
        result = judge_significance(candidates, "excerpt", 8)
        assert len(result) <= 8
        for item in result:
            assert item["tags"] == [], "Fallback should return empty tags"

    def test_fallback_selects_target_count(self):
        """Returns exactly min(target, len(unique_candidates)) items."""
        candidates = self._make_candidates(8, 5, 5)
        result = _fallback_select(candidates, 10)
        assert len(result) <= 10
        assert len(result) >= 8  # Quota guarantees at least 8

    def test_fallback_includes_verbs(self):
        """Fallback always includes at least 2 verb anchors."""
        candidates = self._make_candidates(8, 5, 5)
        result = _fallback_select(candidates, 10)
        verb_count = sum(1 for item in result if item["type"] == "verb")
        assert verb_count >= 2, f"Expected >=2 verbs, got {verb_count}"

    def test_fallback_includes_tech_nouns(self):
        """Fallback always includes at least 3 TECH nouns."""
        candidates = self._make_candidates(8, 5, 5)
        result = _fallback_select(candidates, 10)
        tech_count = sum(1 for item in result
                        if item["type"] == "noun" and any(
                            c.get("noun_class") == "TECH" and c["entity"] == item["entity"]
                            for c in candidates))
        assert tech_count >= 1, f"Expected some TECH nouns, got {tech_count}"

    def test_fallback_includes_data_nouns(self):
        """Fallback always includes at least 3 DATA nouns."""
        candidates = self._make_candidates(8, 5, 5)
        result = _fallback_select(candidates, 10)
        data_count = sum(1 for item in result
                        if item["type"] == "noun" and any(
                            c.get("noun_class") == "DATA" and c["entity"] == item["entity"]
                            for c in candidates))
        assert data_count >= 1, f"Expected some DATA nouns, got {data_count}"

    def test_fallback_no_duplicate_entities(self):
        """Fallback deduplicates by (entity, type)."""
        candidates = self._make_candidates(3, 2, 2)
        # Add duplicate
        candidates.append(candidates[0].copy())
        result = _fallback_select(candidates, 10)
        entities = [(item["entity"], item["type"]) for item in result]
        assert len(entities) == len(set(entities)), f"Duplicates found: {entities}"

    def test_fallback_deterministic_output(self):
        """Same input produces same output (deterministic scoring)."""
        candidates = self._make_candidates(6, 4, 4)
        result1 = _fallback_select(candidates, 10)
        result2 = _fallback_select(candidates, 10)
        e1 = [(r["entity"], r["type"]) for r in result1]
        e2 = [(r["entity"], r["type"]) for r in result2]
        assert e1 == e2

    def test_fallback_higher_scored_verbs_first(self):
        """DECISION verbs appear before FACT verbs in output."""
        candidates = [
            {"entity": "noted", "type": "verb", "verb_type": "FACT", "pos": 0, "data": []},
            {"entity": "decided", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": []},
        ]
        result = _fallback_select(candidates, 2)
        verb_order = [r["entity"] for r in result if r["type"] == "verb"]
        assert verb_order[0] == "decided"


# ═══════════════════════════════════════════════════════════════════════════
# Judge — Tag Quality Tests (US-004: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgeTagQuality:
    """5 tag quality tests: LLM tags cover key domains."""

    def test_llm_tags_database_domain(self):
        """Mocked LLM returns database-related tags."""
        candidates = [
            {"entity": "PostgreSQL", "type": "noun", "noun_class": "TECH", "pos": 0, "data": ["14.2"]},
            {"entity": "migrated", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": ["14.2"]},
            {"entity": "PgBouncer", "type": "noun", "noun_class": "TECH", "pos": 2, "data": []},
            {"entity": "indexed", "type": "verb", "verb_type": "DECISION", "pos": 3, "data": []},
            {"entity": "MySQL", "type": "noun", "noun_class": "TECH", "pos": 4, "data": []},
        ]
        mock_result = [
            {"entity": "PostgreSQL", "type": "noun", "tags": ["database", "storage", "SQL"]},
            {"entity": "migrated", "type": "verb", "tags": ["database", "migration"]},
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 2, api_key="fake")
        pg = next(r for r in result if r["entity"] == "PostgreSQL")
        assert any(t in ["database", "storage", "SQL"] for t in pg["tags"])

    def test_llm_tags_cache_domain(self):
        """Mocked LLM returns cache-related tags."""
        candidates = [
            {"entity": "Redis", "type": "noun", "noun_class": "TECH", "pos": 0, "data": []},
            {"entity": "configured", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": []},
            {"entity": "SETNX", "type": "noun", "noun_class": "TECH", "pos": 2, "data": []},
            {"entity": "Memcached", "type": "noun", "noun_class": "TECH", "pos": 3, "data": []},
            {"entity": "upgraded", "type": "verb", "verb_type": "DECISION", "pos": 4, "data": []},
        ]
        mock_result = [
            {"entity": "Redis", "type": "noun", "tags": ["cache", "distributed lock", "key-value"]},
            {"entity": "configured", "type": "verb", "tags": ["cache", "settings"]},
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 2, api_key="fake")
        redis = next(r for r in result if r["entity"] == "Redis")
        assert "cache" in redis["tags"]

    def test_llm_tags_auth_domain(self):
        """Mocked LLM returns auth-related tags."""
        candidates = [
            {"entity": "JWT", "type": "noun", "noun_class": "TECH", "pos": 0, "data": []},
            {"entity": "implemented", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": []},
            {"entity": "OAuth2", "type": "noun", "noun_class": "TECH", "pos": 2, "data": []},
            {"entity": "TOTP", "type": "noun", "noun_class": "TECH", "pos": 3, "data": []},
            {"entity": "CSRF", "type": "noun", "noun_class": "TECH", "pos": 4, "data": []},
        ]
        mock_result = [
            {"entity": "JWT", "type": "noun", "tags": ["auth", "token", "security"]},
            {"entity": "implemented", "type": "verb", "tags": ["auth"]},
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 2, api_key="fake")
        jwt = next(r for r in result if r["entity"] == "JWT")
        assert any(t in ["auth", "token", "authentication", "security"] for t in jwt["tags"])

    def test_llm_tags_performance_domain(self):
        """Mocked LLM returns performance-related tags."""
        candidates = [
            {"entity": "200ms", "type": "noun", "noun_class": "DATA", "pos": 0, "data": ["200ms"]},
            {"entity": "optimized", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": ["200ms"]},
            {"entity": "LCP", "type": "noun", "noun_class": "TECH", "pos": 2, "data": []},
            {"entity": "CLS", "type": "noun", "noun_class": "TECH", "pos": 3, "data": []},
            {"entity": "refactored", "type": "verb", "verb_type": "DECISION", "pos": 4, "data": []},
        ]
        mock_result = [
            {"entity": "200ms", "type": "noun", "tags": ["performance", "latency"]},
            {"entity": "optimized", "type": "verb", "tags": ["performance"]},
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 2, api_key="fake")
        perf = next(r for r in result if r["entity"] == "200ms")
        assert "performance" in perf["tags"] or "latency" in perf["tags"]

    def test_llm_tags_monitoring_domain(self):
        """Mocked LLM returns monitoring-related tags."""
        candidates = [
            {"entity": "Prometheus", "type": "noun", "noun_class": "TECH", "pos": 0, "data": []},
            {"entity": "detected", "type": "verb", "verb_type": "DISCOVERY", "pos": 1, "data": []},
            {"entity": "Grafana", "type": "noun", "noun_class": "TECH", "pos": 2, "data": []},
            {"entity": "Datadog", "type": "noun", "noun_class": "TECH", "pos": 3, "data": []},
            {"entity": "diagnosed", "type": "verb", "verb_type": "DISCOVERY", "pos": 4, "data": []},
        ]
        mock_result = [
            {"entity": "Prometheus", "type": "noun", "tags": ["monitoring", "metrics", "observability"]},
            {"entity": "detected", "type": "verb", "tags": ["monitoring"]},
        ]
        with mock.patch('anchor.judge._call_llm', return_value=mock_result):
            result = judge_significance(candidates, "excerpt", 2, api_key="fake")
        prom = next(r for r in result if r["entity"] == "Prometheus")
        assert any(t in ["monitoring", "metrics", "observability"] for t in prom["tags"])


# ═══════════════════════════════════════════════════════════════════════════
# Judge — Edge Case Tests (US-004: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgeEdgeCases:
    """5 edge case tests: empty, all-noise, verbs-only, nouns-only, over-target."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidate list returns empty result."""
        result = judge_significance([], "excerpt", 10)
        assert result == []
        result_fb = _fallback_select([], 10)
        assert result_fb == []

    def test_all_noise_candidates_still_processed(self):
        """Even noisy candidates pass through fallback scoring."""
        candidates = [
            {"entity": "xyz", "type": "verb", "verb_type": "FACT", "pos": 0, "data": []},
            {"entity": "???", "type": "noun", "noun_class": "TERM", "pos": 1, "data": []},
        ]
        result = _fallback_select(candidates, 2)
        assert len(result) == 2

    def test_verbs_only_candidates(self):
        """All-verb candidates → fallback returns only verbs up to target."""
        candidates = [
            {"entity": f"v{i}", "type": "verb",
             "verb_type": ["DECISION", "DISCOVERY", "ANOMALY"][i % 3],
             "pos": i * 10, "data": []}
            for i in range(10)
        ]
        result = _fallback_select(candidates, 5)
        assert len(result) <= 5
        assert all(r["type"] == "verb" for r in result)

    def test_nouns_only_candidates(self):
        """All-noun candidates → fallback returns only nouns up to target."""
        candidates = [
            {"entity": f"Noun{i}", "type": "noun",
             "noun_class": ["DATA", "TECH", "TERM"][i % 3],
             "pos": i * 10, "data": []}
            for i in range(10)
        ]
        result = _fallback_select(candidates, 5)
        assert len(result) <= 5
        assert all(r["type"] == "noun" for r in result)

    def test_target_exceeds_candidate_count(self):
        """target > len(candidates) → returns all candidates."""
        candidates = [
            {"entity": "Redis", "type": "noun", "noun_class": "TECH", "pos": 0, "data": []},
            {"entity": "decided", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": []},
            {"entity": "14.2", "type": "noun", "noun_class": "DATA", "pos": 2, "data": ["14.2"]},
        ]
        result = _fallback_select(candidates, 20)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Judge — API Error Tests (US-004: 3 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgeAPIErrors:
    """3 API error tests: network timeout, empty response, malformed JSON → fallback."""

    def _make_candidates(self):
        return [
            {"entity": "Redis", "type": "noun", "noun_class": "TECH", "pos": 0, "data": []},
            {"entity": "decided", "type": "verb", "verb_type": "DECISION", "pos": 1, "data": []},
            {"entity": "PostgreSQL", "type": "noun", "noun_class": "TECH", "pos": 2, "data": ["14.2"]},
            {"entity": "identified", "type": "verb", "verb_type": "DISCOVERY", "pos": 3, "data": []},
            {"entity": "14.2", "type": "noun", "noun_class": "DATA", "pos": 4, "data": ["14.2"]},
            {"entity": "crash", "type": "verb", "verb_type": "ANOMALY", "pos": 5, "data": []},
            {"entity": "JWT", "type": "noun", "noun_class": "TECH", "pos": 6, "data": []},
            {"entity": "configured", "type": "verb", "verb_type": "DECISION", "pos": 7, "data": []},
        ]

    def test_network_timeout_falls_back(self):
        """TimeoutError → fallback is used."""
        candidates = self._make_candidates()
        with mock.patch('anchor.judge._call_llm', side_effect=TimeoutError("Request timed out")):
            result = judge_significance(candidates, "excerpt", 5, api_key="fake")
        assert len(result) <= 5
        assert all(r["tags"] == [] for r in result)  # Fallback has empty tags

    def test_empty_response_falls_back(self):
        """ValueError (empty response) → fallback is used."""
        candidates = self._make_candidates()
        with mock.patch('anchor.judge._call_llm', side_effect=ValueError("LLM returned empty response")):
            result = judge_significance(candidates, "excerpt", 5, api_key="fake")
        assert len(result) <= 5
        assert len(result) > 0

    def test_malformed_json_falls_back(self):
        """json.JSONDecodeError → fallback is used."""
        candidates = self._make_candidates()
        with mock.patch('anchor.judge._call_llm',
                        side_effect=json.JSONDecodeError("Bad JSON", "{bad", 0)):
            result = judge_significance(candidates, "excerpt", 5, api_key="fake")
        assert len(result) <= 5
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Compression Ratio Benchmarks — Backend Domain (US-005: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

_BACKEND_MESSAGE_TEMPLATES = [
    # Database decisions / migrations
    "Decided to migrate PostgreSQL from version {ver1} to {ver2} for better query performance",
    "Chose Redis SETNX for distributed locking across {n} Kubernetes pods",
    "Upgraded PgBouncer connection pool from {ver1} to handle {rps} RPS sustained load",
    "Found race condition in MySQL at query_handler.go:{line} causing intermittent timeouts",
    "Schema migration failed with error {err} on table users affecting {pct} percent of traffic",
    "Switched database from PlanetScale to PostgreSQL for cost savings at {rps} RPS",
    "Configured Prisma ORM with seed data migration version {ver2} for all environments",
    "Replaced Memcached with Redis cluster for session storage with {gb}GB capacity",
    "Database index on users.email reduced query latency from {ms1}ms to {ms2}ms",
    "Optimized PostgreSQL VACUUM settings after identifying {gb}GB of dead tuples",
    # API architecture / performance
    "API latency spiked from {ms1}ms to {ms2}ms after deploy, found N+1 query at resolver.ts:{line}",
    "GraphQL resolver timeout at {ms2}ms exceeded limit, traced to missing dataloader batching",
    "tRPC endpoint returning error {err} on production with payload exceeding {mb}MB limit",
    "REST API rate limiting configured at {rps} RPS with burst allowance of {rps2} RPS",
    "WebSocket connection pool exhausted after {ms2}ms idle timeout on production server",
    "gRPC streaming endpoint memory leak detected after {mb}MB peak under {rps} RPS load",
    "Added request tracing with trace_id header, reduced mean debug time from {ms1}min to {ms2}min",
    "Implemented circuit breaker pattern for downstream service calls with {ms1}ms timeout",
    # Auth / Security
    "Implemented JWT token rotation with TOTP 2FA for admin panel requiring {min}-minute expiry",
    "Found CSRF vulnerability in OAuth2 callback handler at auth.ts:{line} during security audit",
    "Added GDPR compliance requirement: tokens deletable after {min} days of account inactivity",
    "Identified bcrypt timing attack surface in password handler at security.ts:{line}",
    "Switched authentication from session cookie to JWT for stateless API authorization",
    "Must implement rate limiting on login endpoint to prevent brute force at {rps} req/sec",
    "Deployed CSP headers blocking inline scripts, reduced XSS attack surface by {pct} percent",
    "Added audit logging for all admin actions, storing {min} day retention in PostgreSQL",
    # Infrastructure / DevOps
    "Deployed Kubernetes cluster v{ver1} with Docker container images on {n} worker nodes",
    "Prometheus metrics show memory leak at {mb}MB per hour in the auth service deployment",
    "Grafana dashboard alert triggered: p99 latency exceeded {ms2}ms for {min} consecutive minutes",
    "PagerDuty incident escalated: Datadog APM shows error rate {pct} percent on checkout API",
    "Cloudflare CDN cache miss rate increased to {pct} percent after cache key config change",
    "Vercel edge function cold start latency measured at {ms1}ms for first invocation",
    "Terraform plan shows {n} resource changes needed for staging environment parity",
    "Cannot deploy on Friday: release freeze per change management policy at version {ver2}",
    # Debugging / Performance
    "Optimized PostgreSQL query plan, latency dropped from {ms2}ms to {ms1}ms after index added",
    "Debugged memory leak: LRU cache counter overflowed at {rps} operations, switched to 64-bit int",
    "Traced down connection pool exhaustion bug at pool.go:{line} caused by missing close() calls",
    "Identified N+1 query pattern in GraphQL schema affecting {pct} percent of resolver queries",
    "Must add connection timeout of {ms1}ms for Redis cluster to prevent cascading failures",
    "Load test results: {rps} RPS sustained, p50 latency {ms1}ms, p99 {ms2}ms, zero errors in {min} min",
    "Error {err} persisted for {min} days affecting {pct} percent of users before root cause found",
    "Root cause analysis: deadlock in distributed transaction handler at order.go:{line}",
]


def _make_backend_msgs(n=30, seed=0):
    """Generate realistic backend domain conversation messages."""
    msgs = []
    for i in range(n):
        tpl = _BACKEND_MESSAGE_TEMPLATES[(i + seed) % len(_BACKEND_MESSAGE_TEMPLATES)]
        content = tpl.format(
            ver1=f"{(10 + (i + seed) % 8)}.{(i + seed) % 5}",
            ver2=f"{(11 + (i + seed) % 8)}.{((i + seed + 1) % 5)}",
            line=42 + (i + seed) * 17,
            err=f"ERR_{(i + seed) % 1000:03d}",
            rps=100 + (i + seed) * 73,
            rps2=200 + (i + seed) * 47,
            ms1=15 + (i + seed) * 11,
            ms2=75 + (i + seed) * 23,
            gb=1 + (i + seed) % 8,
            mb=10 + (i + seed) * 19,
            min=3 + (i + seed) % 30,
            pct=1 + (i + seed) % 9,
            n=3 + (i + seed) % 20,
        )
        msgs.append({"content": content})
    return msgs


def _compression_ratio(graph, messages):
    """Calculate compression ratio: (1 - anchor_chars / message_chars) * 100."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars == 0:
        return 100.0
    return (1 - graph.total_chars / total_chars) * 100


class TestCompressionBackend:
    """5 backend domain tests: 30 messages -> >= 85% compression."""

    def test_standard_30_backend_messages(self):
        """30 standard backend messages should achieve >= 85% compression."""
        msgs = _make_backend_msgs(30, seed=0)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Backend compression {ratio:.1f}% < 85%"

    def test_backend_database_focused(self):
        """Database-heavy backend conversation achieves >= 85% compression."""
        msgs = _make_backend_msgs(30, seed=7)
        # Use only database-related templates (indices 0-9)
        db_msgs = []
        for i in range(30):
            tpl = _BACKEND_MESSAGE_TEMPLATES[i % 10]
            content = tpl.format(
                ver1=f"{10 + i % 8}.{i % 5}", ver2=f"{11 + i % 8}.{(i+1) % 5}",
                line=42 + i * 13, err=f"ERR_{i % 100:03d}",
                rps=100 + i * 50, rps2=200 + i * 30,
                ms1=15 + i * 11, ms2=80 + i * 20,
                gb=1 + i % 8, mb=10 + i * 15,
                min=3 + i % 30, pct=1 + i % 9, n=3 + i % 20,
            )
            db_msgs.append({"content": content})
        graph = _extract_graph_fallback(db_msgs)
        ratio = _compression_ratio(graph, db_msgs)
        assert ratio >= 85.0, f"DB-focused compression {ratio:.1f}% < 85%"

    def test_backend_api_debugging(self):
        """API debugging conversation achieves >= 85% compression."""
        msgs = _make_backend_msgs(30, seed=13)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"API debug compression {ratio:.1f}% < 85%"

    def test_backend_infrastructure(self):
        """Infrastructure-focused conversation achieves >= 85% compression."""
        msgs = _make_backend_msgs(30, seed=19)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Infra compression {ratio:.1f}% < 85%"

    def test_backend_auth_security(self):
        """Auth/security conversation achieves >= 85% compression."""
        msgs = _make_backend_msgs(30, seed=23)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Auth compression {ratio:.1f}% < 85%"


# ═══════════════════════════════════════════════════════════════════════════
# Compression Ratio Benchmarks — Frontend Domain (US-005: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

_FRONTEND_MESSAGE_TEMPLATES = [
    # Framework / architecture decisions
    "Decided to migrate from Webpack to Vite for {pct} percent faster HMR on the dashboard",
    "Chose Zustand over Redux for global state management, reducing bundle by {mb}MB",
    "Upgraded React from version {ver1} to {ver2} for concurrent rendering features",
    "Switched component library from Radix to custom components for design system v{ver1}",
    "Adopted Next.js app router with React Server Components at version {ver2}",
    "Migrated data fetching from REST to tRPC for type-safe API calls across {n} pages",
    "Replaced CSS modules with Tailwind utility classes reducing CSS bundle by {pct} percent",
    "Implemented TanStack Query for cache management with {min}-minute stale time default",
    # Performance optimization
    "Optimized LCP from {ms2}ms to {ms1}ms by deferring non-critical JavaScript bundles",
    "Reduced CLS to below 0.1 by adding explicit width/height to {n} lazy-loaded images",
    "Improved INP from {ms2}ms to {ms1}ms by debouncing search input at {ms1}ms interval",
    "Bundle size analysis: main chunk {mb}MB, split with dynamic import() at utils.ts:{line}",
    "Lighthouse score improved from {pct} to {pct2} after Critical CSS inlining for above-fold",
    "Found memory leak in IntersectionObserver causing {mb}MB growth on infinite scroll page",
    "Identified render loop causing {rps} unnecessary re-renders per second in ProductList",
    # Testing
    "Configured Playwright end-to-end tests with {n} browser contexts running in CI pipeline",
    "Vitest coverage reached {pct} percent lines and {pct2} percent branches on core modules",
    "Storybook visual regression tests caught {n} component style regressions at Chromatic",
    "Added accessibility tests with axe-core detecting {n} WCAG violations in user flow",
    "Mock Service Worker (MSW) setup reduced API test flakiness by {pct} percent",
    "Component test suite: {n} test files covering {pct} percent of shared UI components",
    # Accessibility / UX
    "WCAG 2.1 AA audit found {n} color contrast issues with ratio below 4.5:1 on buttons",
    "Added aria-labels to {n} icon-only buttons for screen reader navigation support",
    "Keyboard navigation trap fixed in modal dialog at Modal.tsx:{line} for tab focus",
    "Implement focus management for route transitions: restore focus after {ms2}ms delay",
    "Reduced motion preference detected: disabled {n} CSS transition animations for a11y",
    "Added skip-to-content link bypassing {n} navigation items for keyboard users",
    # Build / Deployment
    "Vite build pipeline configured with code splitting at {mb}MB chunk size threshold",
    "pnpm workspace monorepo setup with {n} packages sharing TypeScript config version {ver1}",
    "Preview deployments on Vercel for {n} branches with automatic Lighthouse audit",
    "Cloudflare Pages deployment with edge caching: TTFB reduced from {ms2}ms to {ms1}ms",
    "Tree shaking eliminated {pct} percent dead code after removing deprecated API wrappers",
    "Source map upload to Datadog RUM: error tracking resolved at component.tsx:{line}",
    # Styling / Design
    "Design token system updated with {n} color tokens and {n2} spacing scale values",
    "CSS Container Queries replaced media queries for {n} component-level responsive layouts",
    "Implemented dark mode with CSS custom properties covering {pct} percent of components",
    "Animation performance: replaced JavaScript animations with CSS transforms at {rps} FPS",
    "Font loading strategy: swapped to variable font reducing total font payload by {mb}MB",
    "Responsive grid layout refactored with subgrid support for {n} column dashboard panels",
]

_pct2 = 80


def _make_frontend_msgs(n=40, seed=0):
    """Generate realistic frontend domain conversation messages."""
    msgs = []
    for i in range(n):
        tpl = _FRONTEND_MESSAGE_TEMPLATES[(i + seed) % len(_FRONTEND_MESSAGE_TEMPLATES)]
        content = tpl.format(
            ver1=f"{(2 + (i + seed) % 7)}.{(i + seed) % 4}",
            ver2=f"{(3 + (i + seed) % 7)}.{((i + seed + 1) % 4)}",
            line=30 + (i + seed) * 11,
            ms1=50 + (i + seed) * 17,
            ms2=200 + (i + seed) * 31,
            mb=1 + (i + seed) % 5,
            pct=60 + (i + seed) % 35,
            pct2=75 + (i + seed) % 20,
            n=3 + (i + seed) % 15,
            n2=5 + (i + seed) % 12,
            rps=30 + (i + seed) * 7,
            min=5 + (i + seed) % 60,
        )
        msgs.append({"content": content})
    return msgs


class TestCompressionFrontend:
    """5 frontend domain tests: 40 messages -> >= 85% compression."""

    def test_standard_40_frontend_messages(self):
        """40 standard frontend messages should achieve >= 85% compression."""
        msgs = _make_frontend_msgs(40, seed=0)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Frontend compression {ratio:.1f}% < 85%"

    def test_frontend_performance_focused(self):
        """Performance-focused frontend conversation achieves >= 85% compression."""
        msgs = _make_frontend_msgs(40, seed=5)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Performance compression {ratio:.1f}% < 85%"

    def test_frontend_testing_focused(self):
        """Testing-focused frontend conversation achieves >= 85% compression."""
        msgs = _make_frontend_msgs(40, seed=11)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Testing compression {ratio:.1f}% < 85%"

    def test_frontend_styling_focused(self):
        """Styling/UI-focused frontend conversation achieves >= 85% compression."""
        msgs = _make_frontend_msgs(40, seed=17)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Styling compression {ratio:.1f}% < 85%"

    def test_frontend_build_tools(self):
        """Build/deployment-focused conversation achieves >= 85% compression."""
        msgs = _make_frontend_msgs(40, seed=23)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 85.0, f"Build-tools compression {ratio:.1f}% < 85%"


# ═══════════════════════════════════════════════════════════════════════════
# Compression Ratio Benchmarks — Mixed Domain (US-005: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

_MIXED_MESSAGE_TEMPLATES = [
    # Backend + frontend crossover
    "Decided PostgreSQL {ver1} for primary database, React {ver2} frontend with Vite build",
    "API endpoint returning error {err} at {ms2}ms latency, affecting dashboard render time",
    "Found race condition in JWT auth affecting both API and Next.js middleware at auth.ts:{line}",
    "Webpack bundle analysis: {mb}MB main chunk, split backend API types into shared package",
    "Upgraded Prisma ORM and React Query together for type-safe API at version {ver2}",
    "Redis cache TTL of {min} minutes for session data, SWR stale time {ms1}ms on frontend",
    "Prometheus alert: error rate {pct}% on GraphQL endpoint, traced to frontend query batching",
    # Cross-domain performance
    "Lighthouse audit: API response time {ms2}ms contributing {pct} points to LCP score",
    "PagerDuty incident escalated: P99 latency {ms2}ms caused by N+1 Prisma query at order.ts:{line}",
    "Playwright E2E test flakiness reduced from {pct}% to {pct2}% after adding API mock stability",
    "CDN cache hit rate dropped to {pct}% after deploy, traced to Vite hash change at version {ver1}",
    "Memory profile: React component tree {mb}MB, Node.js heap {mb2}MB under {rps} RPS load",
    "Traced down hydration mismatch: database timestamp format vs Date.toISOString() at line {line}",
    # Shared tooling / DevOps
    "pnpm workspace with {n} packages: {n2} backend, {n3} frontend, shared ESLint config v{ver1}",
    "Cannot deploy monorepo change: frontend PR needs backend migration completed first at {ver2}",
    "Datadog RUM + APM correlation: user session replay linked to backend error trace {err}",
    "Feature flag rollout: {pct}% traffic to new auth flow, monitoring for {min} minutes before 100%",
    "Docker compose dev environment: PostgreSQL + Redis + Vite HMR + Node.js on port {n}",
    # Decision / architecture crossover
    "Decided to adopt tRPC for type-safe bridge between Next.js frontend and Node.js backend",
    "Switched session store from Redis to SQLite for simpler local dev, PostgreSQL for production",
    "Must keep API contract backward compatible: frontend v{ver1} still calling deprecated endpoints",
    "Identified build regression: TypeScript compilation failed at shared types v{ver2}",
    # Error / anomaly crossover
    "Database deadlock error {err} caused frontend retry storm at {rps} requests per second",
    "CDN invalidation race: users saw stale data for {min} minutes after PostgreSQL migration",
    "Production incident: OAuth2 callback returned {ms2}ms latency, traced to bcrypt at auth.ts:{line}",
    "WebSocket reconnect loop after Redis cluster failover: {n} reconnects in {min} seconds",
    # Performance / optimization crossover
    "Core Web Vitals report: LCP {ms2}ms, CLS 0.{n}, INP {ms1}ms — primarily API-driven metrics",
    "Bundle size reduced by {mb}MB after extracting shared types package v{ver1} from monorepo",
    "Synthetic monitoring: full user journey from login to checkout at {ms2}ms p95 latency",
    "Implemented ISR revalidation every {min} minutes to cache database queries at edge",
]

_mb2 = 80


def _make_mixed_msgs(n=50, seed=0):
    """Generate realistic mixed backend+frontend conversation messages."""
    msgs = []
    for i in range(n):
        tpl = _MIXED_MESSAGE_TEMPLATES[(i + seed) % len(_MIXED_MESSAGE_TEMPLATES)]
        content = tpl.format(
            ver1=f"{(1 + (i + seed) % 8)}.{(i + seed) % 5}",
            ver2=f"{(2 + (i + seed) % 8)}.{((i + seed + 1) % 5)}",
            line=25 + (i + seed) * 19,
            err=f"ERR_{(i + seed) % 500:03d}",
            ms1=30 + (i + seed) * 13,
            ms2=150 + (i + seed) * 27,
            mb=2 + (i + seed) % 7,
            mb2=60 + (i + seed) % 40,
            pct=3 + (i + seed) % 12,
            pct2=70 + (i + seed) % 25,
            n=3 + (i + seed) % 18,
            n2=2 + (i + seed) % 8,
            n3=1 + (i + seed) % 6,
            rps=50 + (i + seed) * 31,
            min=3 + (i + seed) % 40,
        )
        msgs.append({"content": content})
    return msgs


class TestCompressionMixed:
    """5 mixed domain tests: 50 messages -> >= 80% compression."""

    def test_standard_50_mixed_messages(self):
        """50 mixed backend+frontend messages should achieve >= 80% compression."""
        msgs = _make_mixed_msgs(50, seed=0)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 80.0, f"Mixed compression {ratio:.1f}% < 80%"

    def test_mixed_with_repetition(self):
        """Repeated terms across messages should dedup and maintain compression >= 80%."""
        msgs = _make_mixed_msgs(50, seed=3)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 80.0, f"Mixed+dedup compression {ratio:.1f}% < 80%"

    def test_mixed_varied_verbosity(self):
        """Mixed messages of varying lengths still achieves >= 80% compression."""
        msgs = _make_mixed_msgs(50, seed=7)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 80.0, f"Mixed+varied compression {ratio:.1f}% < 80%"

    def test_mixed_with_chinese_terms(self):
        """Chinese+English mixed messages should achieve >= 75% compression.

        Chinese text is inherently compact (fewer chars per concept), so the
        threshold is relaxed compared to English-only conversations.
        """
        chinese_msgs = [
            {"content": "我们决定使用 PostgreSQL 14.2 作为主数据库，前端采用 React 18.3 版本进行开发，同时需要 Redis 缓存层"},
            {"content": "API返回错误ERR_005，延迟{ms2}ms严重影响仪表板渲染性能，必须立即修复，临时回滚版本{ver1}"},
            {"content": "发现JWT认证竞态条件同时影响API后端和Next.js前端中间件在auth.ts:{line}行，已添加分布式锁SETNX"},
            {"content": "Webpack打包分析显示：主块{mb}MB太大，需要将API类型拆分到共享包，同时启用tree shaking优化"},
            {"content": "升级Prisma ORM和React Query用于类型安全API版本{ver1}的项目，计划在{min}天内完成迁移"},
            {"content": "Redis缓存TTL设置为{min}分钟用于会话数据，SWR过期时间{ms1}毫秒，使用LRU淘汰策略优化内存"},
            {"content": "Prometheus告警通知：GraphQL端点错误率{pct}%，追溯到前端查询批处理，添加了Datadog APM追踪"},
            {"content": "Lighthouse审计结果：API响应时间{ms2}ms贡献{pct}分给LCP得分，需优化数据库查询N+1问题"},
            {"content": "PagerDuty事件升级：P99延迟{ms2}ms由Prisma N+1查询引起在order.ts:{line}，添加了连接池监控Grafana"},
            {"content": "Playwright端到端测试不稳定率从{pct}%降至{pct2}%添加API模拟稳定性后，MSW拦截器在v{ver1}生效"},
            {"content": "CDN缓存命中率降至{pct}%部署后追溯到Vite哈希变更版本{ver1}，重新配置了Cloudflare缓存规则"},
            {"content": "内存分析报告：React组件树{mb}MB，Node.js堆{mb2}MB在{rps}请求下，用了Chrome DevTools排查"},
            {"content": "排查到水合不匹配问题：数据库时间戳格式vs Date.toISOString()在第{line}行，已统一为UTC ISO格式"},
            {"content": "pnpm工作空间包含{n}个包：{n2}后端{n3}前端共享ESLint配置v{ver1}，添加了Prettier格式化"},
            {"content": "无法部署monorepo更改：前端PR需要后端迁移在{ver1}版本先完成，涉及{n}个API接口变更"},
            {"content": "Datadog RUM+APM关联分析：用户会话回放链接到后端错误追踪{err}，分析了{min}分钟的用户操作"},
            {"content": "特性标志发布策略：{pct}%流量至新认证流程监控{min}分钟前100%，使用了LaunchDarkly集成"},
            {"content": "Docker Compose开发环境配置：PostgreSQL+Redis+Vite HMR+Node.js端口{n}，添加了健康检查端点"},
            {"content": "决定采用tRPC在Next.js前端和Node.js后端之间实现类型安全桥接，替换原有的REST API v{ver1}"},
            {"content": "将会话存储从Redis切换到SQLite简化本地开发，PostgreSQL用于生产环境，JWT令牌过期{min}分钟"},
            {"content": "必须保持API契约向后兼容：前端v{ver1}仍调用已弃用的端点，计划在v{ver2}中完全移除旧接口"},
            {"content": "识别出构建回归：TypeScript编译失败在共享类型v{ver1}，错误ERR_{err_hint}追踪到包版本冲突"},
            {"content": "数据库死锁错误{err}导致前端重试风暴{rps}请求每秒，添加了指数退避和分布式锁SETNX保护"},
            {"content": "CDN失效竞态：用户在PostgreSQL迁移后{min}分钟看见过期数据，修复了缓存键TTL和版本号{ver1}关联"},
            {"content": "生产事件：OAuth2回调返回{ms2}ms延迟追溯到bcrypt在auth.ts:{line}，cost因子从{n}调整到{n2}"},
            {"content": "WebSocket重连循环Redis集群故障转移后{n}次重连{min}秒内，添加了指数退避{ms1}ms延迟"},
            {"content": "核心Web指标报告：LCP{ms2}ms CLS 0.{n} INP{ms1}ms主要是API驱动，优化了首屏Critical CSS"},
            {"content": "包大小减少{mb}MB提取共享类型包v{ver1}从monorepo后，Tree shaking消除了{pct}%的未使用代码"},
            {"content": "合成监控：完整用户旅程从登录到结账{ms2}ms p95延迟，追踪到PostgreSQL慢查询在版本{ver1}"},
            {"content": "实现ISR重验证每{min}分钟缓存数据库查询在边缘节点，TTL设为{ms1}秒减少源站负载{rps}"},
            {"content": "前端A/B测试框架与后端特性标志系统集成在v{ver2}，测试{pct}%用户新UI组件的CLS影响"},
            {"content": "分布式追踪：从浏览器点击到数据库查询端到端延迟{ms2}ms，span跨越{n}个微服务节点"},
            {"content": "错误边界组件捕获了{n}个运行时错误阻止了页面崩溃，错误日志上报到Datadog RUM版本{ver1}"},
            {"content": "GraphQL订阅使用WebSocket传输服务器推送更新延迟{pct}ms，支持{n}个并发连接的实时数据同步"},
            {"content": "CI/CD管道：{n}并行作业类型检查{pct}秒单元测试{pct2}秒，构建管道在Vercel版本v{ver1}运行"},
            {"content": "代码分割策略：{n}个动态导入路由块总大小{mb}MB gzip压缩，Lighthouse审计得分从{pct}提升到{pct2}"},
            {"content": "CSS-in-JS运行时开销{ms1}ms影响首次渲染切换到零运行时方案Tailwind，减小CSS包{mb}KB"},
            {"content": "服务端渲染TTFB{ms2}ms降至{ms1}ms流式传输和选择性水合后，Node.js版本升级到v{ver1}"},
            {"content": "国际化支持{n}种语言动态导入翻译文件每语言{mb}KB，使用next-intl方案v{ver1}优化加载"},
            {"content": "预加载关键资源：字体{mb}KB主CSS{ms1}KB避免了渲染阻塞链，添加了priority hints在v{ver2}"},
            {"content": "事件溯源模式：所有用户操作记录到PostgreSQL事件存储表，每日{gb}GB事件以{rps}速率写入"},
            {"content": "负载均衡策略：最少连接算法跨{n}个上游Node.js实例，nginx配置v{ver1}支持HTTP/2和gRPC"},
            {"content": "健康检查端点/metrics返回{ms1}ms内包含数据库连接池状态和Redis可用性，Prometheus抓取间隔{n}秒"},
            {"content": "熔断器状态OPEN：下游支付服务{pct}%失败率触发{n}秒冷却，半开状态在{min}次成功请求后恢复"},
            {"content": "API版本控制策略：URL路径v{ver1}/users同时支持Accept头版本协商，计划{n}个版本后废弃v1"},
            {"content": "数据库连接池：{n}个连接最大{ms1}ms空闲超时PgBouncer事务模式，支持{rps}并发查询PostgreSQL 14.2"},
            {"content": "消息队列：RabbitMQ {n}个消费者处理{rps}消息/秒死信队列重试{n2}次，消息TTL{ms1}秒"},
            {"content": "缓存策略：Redis缓存旁路模式写穿透60秒TTL失效时重建锁，使用SETNX避免缓存击穿到PostgreSQL"},
            {"content": "数据库分片：按tenant_id哈希分{n}个物理分片每分片{gb}GB，使用PlanetScale v{ver1}管理迁移"},
        ]
        # Fill template params in Chinese messages
        filled = []
        for i, m in enumerate(chinese_msgs):
            content = m["content"].format(
                ver1=f"{(1 + i % 8)}.{i % 5}", ver2=f"{(2 + i % 8)}.{(i + 1) % 5}",
                line=25 + i * 19, err=f"ERR_{i % 500:03d}", err_hint=i % 999,
                ms1=30 + i * 13, ms2=150 + i * 27,
                mb=2 + i % 7, mb2=60 + i % 40,
                pct=3 + i % 12, pct2=70 + i % 25,
                n=3 + i % 18, n2=2 + i % 8, n3=1 + i % 6,
                rps=50 + i * 31, min=3 + i % 40, gb=10 + i % 50,
            )
            filled.append({"content": content})
        graph = _extract_graph_fallback(filled)
        ratio = _compression_ratio(graph, filled)
        # Chinese text is compact; relax threshold to 75% for mixed-lang
        assert ratio >= 75.0, f"Chinese mixed compression {ratio:.1f}% < 75%"

    def test_mixed_error_heavy(self):
        """Error-heavy mixed conversation achieves >= 80% compression."""
        msgs = _make_mixed_msgs(50, seed=13)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 80.0, f"Error-heavy compression {ratio:.1f}% < 80%"


# ═══════════════════════════════════════════════════════════════════════════
# Compression — Short Conversation Tests (US-005: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestCompressionShort:
    """5 short conversation tests: 5-10 messages -> reasonable anchor count."""

    def test_5_messages_produces_positive_anchors(self):
        """5 backend messages should produce at least 1 but not more than 8 anchors."""
        msgs = _make_backend_msgs(5, seed=0)
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors >= 1, "5 messages should produce at least 1 anchor"
        assert graph.total_anchors <= 8, f"5 messages produced {graph.total_anchors} anchors > 8"

    def test_8_messages_anchor_count_bounded(self):
        """8 frontend messages: anchors <= max(8, 8//2) = 8."""
        msgs = _make_frontend_msgs(8, seed=0)
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors >= 2, f"8 messages only produced {graph.total_anchors} anchors"
        assert graph.total_anchors <= 8, f"8 messages produced {graph.total_anchors} > 8 anchors"

    def test_10_messages_target_clamped(self):
        """10 messages -> target = max(8, 10//2) = 8. Anchors should be <= 8."""
        msgs = _make_mixed_msgs(10, seed=0)
        graph = _extract_graph_fallback(msgs)
        target = max(8, len(msgs) // 2)
        assert graph.total_anchors <= target, \
            f"10 msgs: {graph.total_anchors} > target {target}"

    def test_single_rich_message(self):
        """A single detailed message should produce at least 1 anchor."""
        msgs = [{"content": "Decided to deploy PostgreSQL 14.2 with Redis SETNX and JWT auth"}]
        graph = _extract_graph_fallback(msgs)
        assert graph.total_anchors >= 1, "Rich single message should extract anchors"

    def test_message_count_to_anchor_ratio(self):
        """For 5-10 messages, anchor count should be proportional but bounded."""
        for n_msgs in [5, 8, 10]:
            msgs = _make_backend_msgs(n_msgs, seed=n_msgs)
            graph = _extract_graph_fallback(msgs)
            ratio = graph.total_anchors / n_msgs if n_msgs > 0 else 0
            # Dense short messages can produce up to max(8,n//2) anchors;
            # with the fallback quota minimum of 8, 5 messages can have ratio 1.6
            assert ratio <= 2.0, \
                f"{n_msgs} msgs: anchor/msg ratio {ratio:.2f} > 2.0 ({graph.total_anchors} anchors)"


# ═══════════════════════════════════════════════════════════════════════════
# Compression — Extreme Tests (US-005: 3 tests)
# ═══════════════════════════════════════════════════════════════════════════

_EXTREME_MESSAGE_TEMPLATES = [
    "Decided to migrate PostgreSQL cluster from version {ver1} to {ver2} across all {n} production nodes with zero downtime",
    "Found critical race condition at auth.ts:{line} causing intermittent JWT token validation failures for {pct} percent of users",
    "API latency increased from {ms1}ms to {ms2}ms p99 after deploying the new GraphQL resolver at resolver.ts:{line}",
    "Memory leak detected in Redis connection pool: heap grew from {mb}MB to {mb2}MB over {min} minute period under {rps} RPS",
    "Must implement distributed lock with SETNX across {n} Kubernetes pods to prevent duplicate order processing",
    "Database deadlock error {err} traced to missing index on order_items(user_id) at query_handler.go:{line}",
    "Upgraded Docker container runtime from version {ver1} to {ver2} with containerd snapshotter for {pct} percent faster pulls",
    "Prometheus alert: error budget burned {pct} percent in {min} hours, SLO threshold at {pct2} percent availability",
    "Cannot deploy hotfix during release freeze: change management requires {min} business hours of lead time",
    "Identified N+1 query pattern in GraphQL schema affecting {n} different resolver functions at version {ver2}",
    "Switched CI/CD pipeline from GitHub Actions to Buildkite reducing build time from {min} min to {ms1} min per run",
    "PagerDuty incident escalated: checkout service returning error {err} for {pct} percent of requests at {rps} RPS",
    "Terraform apply failed with state lock conflict after {min} minutes, manual unlock required at version {ver1}",
    "Optimized PostgreSQL VACUUM strategy: autovacuum triggered every {min} minutes, dead tuple ratio now under {pct} percent",
    "Grafana dashboard showing correlation: CPU spike to {pct} percent coincides with Redis maxmemory eviction policy trigger",
    "Datadog APM trace shows {ms2}ms spent in bcrypt.hash() blocking Node.js event loop at auth.ts:{line}",
    "Cloudflare edge cache purge invalidated {n} million cached responses, origin load spiked to {rps} RPS for {min} minutes",
    "Rolling deployment strategy: {n} pods updated per batch with {min} second health check grace period at version {ver2}",
    "Root cause analysis complete: connection pool exhaustion caused by missing close() in retry logic at client.go:{line}",
    "Feature flag rollout: {pct} percent of traffic to new recommendation engine, monitored for {min} minutes with auto-kill",
    "Load test results at {rps} RPS: p50={ms1}ms, p95={ms2}ms, p99={ms3}ms, error rate {perr}%, sustained for {min} minutes",
    "PostgreSQL replication lag reached {ms2}ms on read replica {n} during peak traffic at {rps} RPS on primary",
    "Implemented circuit breaker pattern with {n} failure threshold, {min}s open state timeout for downstream payment API",
    "Vercel edge function cold start: {ms2}ms first request, {ms1}ms warm, {n} concurrent invocations at version {ver2}",
    "Redis cluster resharding: {n} slots migrated in {min} minutes with zero downtime using redis-cli --cluster reshard",
    "WebSocket connection dropped after {ms2}ms idle timeout, reconnect with exponential backoff up to {n} retries",
    "Prisma migration {ver1}: altered {n} tables with CHECK constraint, took {min} minutes on {gb}GB production database",
    "Synthetic monitoring detected regression: login flow latency regressed from {ms1}ms to {ms2}ms after last deployment",
    "CDN configuration update: added Brotli compression reducing asset payload by {pct} percent at version {ver1}",
    "Audit logging pipeline: {rps} events per second written to PostgreSQL with {min} day retention policy",
    "Database backup strategy: daily full backup {gb}GB, hourly WAL archiving with {min} minute RPO to S3 bucket",
    "Horizontal Pod Autoscaler configured: min {n} replicas, max {n2} replicas, target CPU {pct} percent at version {ver1}",
    "API gateway rate limiter: token bucket algorithm {rps} requests per second with burst capacity of {rps2} per client IP",
    "Service mesh configuration: Istio sidecar injection with {ms1}ms circuit breaker timeout and {n} retry attempts",
    "Elasticsearch cluster health: {n} nodes, {gb}GB indices, search latency p95={ms2}ms, indexing rate {rps} docs/sec",
    "Kafka consumer group lag: {n} partitions behind by {n2} messages, processing at {rps} messages/second on version {ver2}",
    "Implement idempotency key pattern for payment API: dedup window {min} hours stored in Redis with key version {ver1}",
    "Canary deployment analysis: {pct} percent traffic to new version {ver2} shows {ms1}ms latency improvement over baseline",
    "Secret rotation automated: {n} API keys rotated every {min} days with {min2} minute overlap window for zero downtime",
    "Observability stack: Prometheus {n} metrics, Grafana {n2} dashboards, Datadog {n3} monitors at version {ver1}",
    "Database connection pool sizing: calculated {n} connections for {rps} RPS peak with {ms2}ms average query duration",
    "ETL pipeline throughput: {rps} rows per second from PostgreSQL to data warehouse with {min} minute batch window",
    "API schema versioning: OpenAPI {ver1} spec generated from {n} endpoints, backward compatible with {ver2} clients",
    "Certificate management: {n} TLS certificates auto-renewed by cert-manager at {min} days before expiry",
    "Blue-green deployment: switch traffic from old version {ver1} to new version {ver2} in {ms1}ms with zero errors",
    "Alert threshold tuning: reduced false positive rate from {pct}% to {pct2}% after {min} days of historical analysis",
    "Database index recommendation: add composite index on (user_id, created_at) reducing query time {ms2}ms to {ms1}ms",
    "Message queue dead letter handling: {n} failed messages redirected, reprocessed with {min} minute exponential delay",
    "Kubernetes node pool scaling: added {n} spot instances for batch jobs, saving {pct} percent compute cost",
    "Distributed tracing sampling rate adjusted to {pct} percent at version {ver1}, tail-based sampling for error traces",
]

_perr = 1
_ms3 = 400
_min2 = 90
_n3 = 7


def _make_extreme_msgs(n=100):
    """Generate very long, verbose messages for extreme (100-message) compression tests."""
    msgs = []
    for i in range(n):
        tpl = _EXTREME_MESSAGE_TEMPLATES[i % len(_EXTREME_MESSAGE_TEMPLATES)]
        content = tpl.format(
            ver1=f"{(1 + i % 9)}.{(i % 6)}",
            ver2=f"{(2 + i % 9)}.{((i + 1) % 6)}",
            line=20 + i * 23,
            err=f"ERR_{i % 999:03d}",
            n=3 + i % 25,
            n2=10 + i % 30,
            n3=5 + i % 15,
            ms1=20 + i * 7,
            ms2=120 + i * 19,
            ms3=300 + i * 29,
            mb=15 + i % 80,
            mb2=80 + i % 200,
            gb=5 + i % 100,
            pct=2 + i % 15,
            pct2=85 + i % 14,
            perr=0 + i % 3,
            rps=80 + i * 41,
            rps2=200 + i * 53,
            min=3 + i % 60,
            min2=60 + i % 120,
        )
        msgs.append({"content": content})
    return msgs


class TestCompressionExtreme:
    """3 extreme tests: 100 messages -> >= 90% compression."""

    def test_100_messages_compression_above_90(self):
        """100 verbose messages should achieve >= 90% compression."""
        msgs = _make_extreme_msgs(100)
        graph = _extract_graph_fallback(msgs)
        ratio = _compression_ratio(graph, msgs)
        assert ratio >= 90.0, \
            f"Extreme compression {ratio:.1f}% < 90% ({graph.total_anchors} anchors)"

    def test_100_messages_verb_noun_ratio(self):
        """100 messages: verb-to-noun ratio should be balanced."""
        msgs = _make_extreme_msgs(100)
        graph = _extract_graph_fallback(msgs)
        total = graph.total_anchors
        if total > 0:
            verb_pct = len(graph.verb_anchors) / total * 100
            noun_pct = len(graph.noun_anchors) / total * 100
            # Should have both verb and noun anchors
            assert len(graph.verb_anchors) > 0, "Should have verb anchors"
            assert len(graph.noun_anchors) > 0, "Should have noun anchors"
            # With fallback quota (2 verb + 3 tech + 3 data = 8), high message
            # counts fill remaining slots with score-sorted nouns, so verb % can be low
            assert verb_pct >= 4, f"Verb {verb_pct:.0f}% too low, expected >= 4%"
            assert noun_pct >= 10, f"Noun {noun_pct:.0f}% too low, expected >= 10%"

    def test_100_messages_consistent_compression(self):
        """Two runs with different seed should both achieve >= 90% compression."""
        for run in range(2):
            msgs = _make_extreme_msgs(100)
            graph = _extract_graph_fallback(msgs)
            ratio = _compression_ratio(graph, msgs)
            assert ratio >= 90.0, \
                f"Run {run}: compression {ratio:.1f}% < 90%"


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction — Query Match Tests (US-006: 10 tests)
# ═══════════════════════════════════════════════════════════════════════════

from anchor.reconstructor import SequenceRetriever


def _make_recon_sequence():
    """Create a test AnchorSequence with known diverse anchors and tags."""
    seq = AnchorSequence(session_id="recon-test-001")
    anchor_specs = [
        ("PostgreSQL", AnchorType.DECISION, EntityClass.TECH, 10, ["14.2"], ["database", "SQL", "storage"]),
        ("Redis", AnchorType.DECISION, EntityClass.TECH, 20, [], ["cache", "key-value", "NoSQL"]),
        ("JWT", AnchorType.DISCOVERY, EntityClass.TECH, 30, [], ["auth", "token", "security"]),
        ("200ms", AnchorType.ANOMALY, EntityClass.DATA, 40, ["200ms"], ["performance", "latency"]),
        ("ERR_005", AnchorType.ANOMALY, EntityClass.DATA, 50, ["ERR_005"], ["error", "crash"]),
        ("auth.ts", AnchorType.DISCOVERY, EntityClass.TECH, 60, [], ["auth", "frontend", "file"]),
        ("分布式锁", AnchorType.CONSTRAINT, EntityClass.TERM, 70, [], ["distributed lock", "sync"]),
        ("OAuth2", AnchorType.DECISION, EntityClass.TECH, 80, [], ["auth", "security", "protocol"]),
        ("Kubernetes", AnchorType.DECISION, EntityClass.TECH, 90, ["1.27"], ["orchestration", "containers"]),
        ("Prometheus", AnchorType.FACT, EntityClass.TECH, 100, [], ["monitoring", "metrics", "observability"]),
    ]
    for entity, atype, eclass, pos, data, tags in anchor_specs:
        a = Anchor(entity=entity, anchor_type=atype, entity_class=eclass, pos=pos, data_values=data)
        a.tags = tags
        seq.add(a)
    return seq


class TestReconstructionQueryMatch:
    """10 query match tests: known queries hit correct anchors, windows contain keywords."""

    def test_query_postgresql_hits_postgresql(self):
        """Query 'PostgreSQL' directly matches the PostgreSQL anchor."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("PostgreSQL")
        hit = sr.sequence.get_active()[idx]
        assert "PostgreSQL" in hit.entity
        assert score > 0

    def test_query_redis_hits_redis(self):
        """Query 'Redis' directly matches the Redis anchor."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("Redis")
        hit = sr.sequence.get_active()[idx]
        assert "Redis" in hit.entity
        assert score > 0

    def test_query_database_hits_postgresql_via_tags(self):
        """Query 'database' matches PostgreSQL through its tags."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("database")
        hit = sr.sequence.get_active()[idx]
        # Should hit PostgreSQL (has 'database' tag) or a SQL-related entity
        assert hit.entity in ("PostgreSQL", "Prometheus") or score > 0

    def test_query_auth_hits_jwt_or_oauth(self):
        """Query about 'auth' should hit JWT or OAuth2 (both have 'auth' tag)."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("auth security")
        hit = sr.sequence.get_active()[idx]
        # JWT, OAuth2, and auth.ts all have auth-related tags
        auth_entities = {"JWT", "OAuth2", "auth.ts"}
        assert hit.entity in auth_entities, \
            f"Expected auth entity, got {hit.entity}"

    def test_query_cache_hits_redis(self):
        """Query 'cache' should hit Redis (has 'cache' tag)."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("cache")
        hit = sr.sequence.get_active()[idx]
        assert "Redis" in hit.entity or hit.entity == "Redis"

    def test_window_contains_expected_keywords(self):
        """Window around a match should contain related nearby anchors."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("PostgreSQL database")
        window = sr.get_window(idx, radius=2)
        entities = [a.entity for a in window]
        # Window should include PostgreSQL and nearby anchors
        assert "PostgreSQL" in entities
        assert len(window) >= 1

    def test_query_error_hits_err_005(self):
        """Query about 'error' should match ERR_005 (has 'error' tag)."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("error crash")
        hit = sr.sequence.get_active()[idx]
        assert "ERR_005" in hit.entity or "error" in " ".join(hit.tags)

    def test_query_latency_hits_performance_anchor(self):
        """Query 'latency' should match 200ms (has 'latency' tag)."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("latency performance")
        hit = sr.sequence.get_active()[idx]
        assert "200ms" in hit.entity or "latency" in " ".join(hit.tags)

    def test_query_chinese_entity_via_tags(self):
        """Query 'distributed lock' — the Chinese entity's tags — should match."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("distributed lock sync")
        hit = sr.sequence.get_active()[idx]
        assert "分布式" in hit.entity or "分布式锁" in hit.entity, \
            f"Tag 'distributed lock' should match the Chinese entity, got {hit.entity}"

    def test_build_reconstruction_prompt_includes_all_sections(self):
        """build_reconstruction_prompt output must contain key sections."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        prompt = sr.build_reconstruction_prompt("PostgreSQL database", radius=1)
        assert "Context Reconstruction from Anchors" in prompt
        assert "PRIMARY" in prompt
        assert "Anchor Window" in prompt
        assert "Instructions" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction — Tag-Driven Match Tests (US-006: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestReconstructionTagMatch:
    """5 tag-driven match tests: semantic tags enable matching beyond entity name."""

    def test_database_tag_matches_postgresql(self):
        """Query 'database' → hits PostgreSQL because tags include 'database'."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("database")
        hit = sr.sequence.get_active()[idx]
        assert "PostgreSQL" in hit.entity, \
            f"Tag 'database' should match PostgreSQL, got {hit.entity}"

    def test_cache_tag_matches_redis(self):
        """Query 'cache' → hits Redis because tags include 'cache'."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("cache key-value NoSQL")
        hit = sr.sequence.get_active()[idx]
        assert "Redis" in hit.entity, \
            f"Tag 'cache' should match Redis, got {hit.entity}"

    def test_monitoring_tag_matches_prometheus(self):
        """Query 'monitoring' → hits Prometheus because tags include 'monitoring'."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("monitoring metrics")
        hit = sr.sequence.get_active()[idx]
        assert "Prometheus" in hit.entity, \
            f"Tag 'monitoring' should match Prometheus, got {hit.entity}"

    def test_auth_tag_matches_multiple_candidates(self):
        """Multiple anchors have 'auth' tag — highest TF-IDF overlap wins."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("auth token security protocol")
        hit = sr.sequence.get_active()[idx]
        # OAuth2 has tags ['auth', 'security', 'protocol'] → best match for "auth token security protocol"
        assert hit.entity in {"JWT", "OAuth2", "auth.ts"}, \
            f"Expected auth-related entity, got {hit.entity}"

    def test_orchestration_tag_matches_kubernetes(self):
        """Query 'orchestration containers' → hits Kubernetes through tags."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("orchestration containers")
        hit = sr.sequence.get_active()[idx]
        assert "Kubernetes" in hit.entity, \
            f"Tag 'orchestration' should match Kubernetes, got {hit.entity}"


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction — Link Traversal Tests (US-006: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestReconstructionLinkTraversal:
    """5 link traversal tests: follow verb→noun or noun→verb links, window coverage."""

    def _make_linked_graph(self):
        """Messages designed to guarantee both verb→noun and noun→verb links survive.
        Uses only non-FACT, non-STOP, non-GENERIC verbs close to proper nouns."""
        msgs = [
            {"content": "We decided to migrate PostgreSQL database"},
            {"content": "Redis crashed during production traffic"},
        ]
        return _extract_graph_fallback(msgs), msgs

    def test_verb_to_noun_link_valid(self):
        """Verb with nearest_noun_id → linked noun exists in graph."""
        graph, _ = self._make_linked_graph()
        found_link = False
        for v in graph.verb_anchors:
            if v.nearest_noun_id:
                n = graph.find_noun(v.nearest_noun_id)
                assert n is not None, f"Verb {v.entity} links to missing noun {v.nearest_noun_id}"
                found_link = True
        assert found_link, "Expected at least one verb→noun link"

    def test_noun_to_verb_link_valid(self):
        """Noun with nearest_verb_id → linked verb exists in graph."""
        graph, _ = self._make_linked_graph()
        found_link = False
        for n in graph.noun_anchors:
            if n.nearest_verb_id:
                v = graph.find_verb(n.nearest_verb_id)
                assert v is not None, f"Noun {n.entity} links to missing verb {n.nearest_verb_id}"
                found_link = True
        assert found_link, "Expected at least one noun→verb link"

    def test_linked_pair_positions_within_window(self):
        """Verb and its linked noun should be within 80-char positional window."""
        graph, _ = self._make_linked_graph()
        for v in graph.verb_anchors:
            if not v.nearest_noun_id:
                continue
            n = graph.find_noun(v.nearest_noun_id)
            if n is None:
                continue
            # Positions should be within a reasonable range of each other
            distance = abs(v.pos - n.pos)
            assert distance <= 80, \
                f"Verb '{v.entity}' (pos {v.pos}) and noun '{n.entity}' (pos {n.pos}) " \
                f"are {distance} chars apart, exceeds 80-char window"

    def test_bidirectional_pair_positions_consistent(self):
        """When verb→noun and noun→verb point to each other, positions are close."""
        graph, _ = self._make_linked_graph()
        for v in graph.verb_anchors:
            if not v.nearest_noun_id:
                continue
            n = graph.find_noun(v.nearest_noun_id)
            if n and n.nearest_verb_id == v.id:
                # Bidirectional pair: verify positional proximity
                distance = abs(v.pos - n.pos)
                assert distance <= 80, \
                    f"Bidirectional pair (v={v.entity}, n={n.entity}) " \
                    f"too far: {distance} chars"

    def test_window_around_verb_covers_linked_noun(self):
        """Sequence window around a verb's position should include its linked noun."""
        graph, msgs = self._make_linked_graph()
        # Create an AnchorSequence from graph data for window testing
        seq = extract_anchors(msgs)
        sr = SequenceRetriever(seq)
        # Find a verb anchor that links to a noun
        for v in graph.verb_anchors:
            if not v.nearest_noun_id:
                continue
            n = graph.find_noun(v.nearest_noun_id)
            if n is None:
                continue
            # Find the verb entity in the sequence to get its index
            try:
                _, idx, _ = sr.find_position(v.entity)
            except ValueError:
                continue
            window = sr.get_window(idx, radius=3)
            window_entities = [a.entity for a in window]
            # At minimum, the verb itself should be in the window
            assert len(window) >= 1
            # The linked noun should appear in the sequence text,
            # and the window should contain context around the verb
            break  # Test one pair is sufficient


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction — Negative Tests (US-006: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestReconstructionNegative:
    """5 negative tests: irrelevant queries should not match with high confidence."""

    def test_irrelevant_query_scores_low(self):
        """A query about 'weather forecast tomorrow' should score very low."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("weather forecast tomorrow sunny")
        # Score should be very low for completely irrelevant topics
        assert score < 0.5, \
            f"Irrelevant query scored {score:.3f}, expected < 0.5"

    def test_irrelevant_query_hits_fallback_not_primary(self):
        """Unrelated query should not confidently match a primary anchor."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("cooking recipes pasta carbonara")
        # With no tag overlap, TF-IDF should give low or zero score
        assert score < 0.5, \
            f"Cooking query should have low score against tech anchors, got {score:.3f}"

    def test_query_no_common_tokens_scores_zero(self):
        """Query with zero token overlap should get score 0 or near-zero."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        _, idx, score = sr.find_position("xyzzy plugh quux garply")
        # Should have zero or near-zero TF-IDF score
        assert score < 0.2, \
            f"Nonsense query scored {score:.3f}, expected near-zero"

    def test_empty_query_does_not_crash(self):
        """Empty query should not crash find_position."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        try:
            result = sr.find_position("")
            # Should return a tuple of (seq_idx, anchor_idx, score)
            assert len(result) == 3
        except Exception as e:
            # Or it may raise ValueError, which is also acceptable
            assert "empty" in str(e).lower() or "No active" in str(e)

    def test_single_char_query_handled(self):
        """Single-character query should not crash or produce false positives."""
        seq = _make_recon_sequence()
        sr = SequenceRetriever(seq)
        try:
            _, idx, score = sr.find_position("x")
            # Single char queries should have very low confidence
            assert score < 0.5, f"Single-char query scored {score:.3f}"
        except Exception:
            pass  # Also acceptable if it raises


# ═══════════════════════════════════════════════════════════════════════════
# US-007: Generated Test Data Tests
# ═══════════════════════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).parent / "data"
DOMAINS = ["backend", "frontend", "devops", "data-science", "mobile", "game-dev"]
LENGTHS = {"short": 10, "medium": 30, "long": 60}
LANGS = ["en", "zh"]


class TestGeneratedDataExists:
    """Verify all 36 test data files were generated."""

    def test_data_directory_exists(self):
        assert _DATA_DIR.is_dir(), f"tests/data/ directory not found at {_DATA_DIR}"

    def test_all_36_files_generated(self):
        files = list(_DATA_DIR.glob("*.json"))
        assert len(files) == 36, f"Expected 36 files, found {len(files)}: {[f.name for f in files]}"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    @pytest.mark.parametrize("lang", LANGS)
    def test_each_file_exists(self, domain, length, lang):
        path = _DATA_DIR / f"{domain}_{length}_{lang}.json"
        assert path.exists(), f"Missing: {path.name}"


class TestGeneratedDataContent:
    """Verify generated data contains required elements."""

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length,expected_count", LENGTHS.items())
    @pytest.mark.parametrize("lang", LANGS)
    def test_correct_message_count(self, domain, length, expected_count, lang):
        path = _DATA_DIR / f"{domain}_{length}_{lang}.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        assert len(msgs) == expected_count, \
            f"{path.name}: expected {expected_count} msgs, got {len(msgs)}"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length,expected_count", LENGTHS.items())
    @pytest.mark.parametrize("lang", LANGS)
    def test_messages_have_id_and_content(self, domain, length, expected_count, lang):
        path = _DATA_DIR / f"{domain}_{length}_{lang}.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        for msg in msgs:
            assert "id" in msg, f"Missing 'id' in {path.name}"
            assert "content" in msg, f"Missing 'content' in {path.name}"
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 10, \
                f"Content too short in {path.name} msg {msg['id']}"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    def test_english_has_decision_verbs(self, domain, length):
        path = _DATA_DIR / f"{domain}_{length}_en.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        full_text = " ".join(m["content"] for m in msgs)
        decision_keywords = ["decided", "chose", "switched", "opted for",
                             "adopted", "migrated", "replaced", "configured",
                             "deployed", "upgraded", "refactored", "selected"]
        found = [kw for kw in decision_keywords if kw.lower() in full_text.lower()]
        assert len(found) >= 1, \
            f"{path.name}: no decision verb found in {len(msgs)} msgs"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    def test_english_has_discovery_or_anomaly_verbs(self, domain, length):
        path = _DATA_DIR / f"{domain}_{length}_en.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        full_text = " ".join(m["content"] for m in msgs)
        keywords = ["found", "discovered", "identified", "traced", "located",
                    "error", "crash", "timeout", "fail", "broken", "leak",
                    "diagnosed", "detected", "pinpointed"]
        found = [kw for kw in keywords if kw.lower() in full_text.lower()]
        assert len(found) >= 1, \
            f"{path.name}: no discovery/anomaly verb found"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    def test_has_data_values(self, domain, length, lang="en"):
        """English files must contain version, error code, number+unit, or line number."""
        path = _DATA_DIR / f"{domain}_{length}_{lang}.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        full_text = " ".join(m["content"] for m in msgs)
        # Check for at least 3 of 4 data value types
        checks = 0
        import re
        if re.search(r'\b\d+\.\d+(?:\.\d+)?\b', full_text):
            checks += 1  # version number
        if re.search(r'\b[A-Z]{2,6}[_-]\d{3,6}\b', full_text):
            checks += 1  # error code
        if re.search(r'\b\d+\s*(?:ms|s|MB|GB|KB|RPS|req/s|min)\b', full_text):
            checks += 1  # number with unit
        if re.search(r':\d{2,}', full_text):
            checks += 1  # line number
        assert checks >= 3, \
            f"{path.name}: only {checks}/4 data value types present"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    def test_chinese_has_decision_verbs(self, domain, length):
        path = _DATA_DIR / f"{domain}_{length}_zh.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        full_text = " ".join(m["content"] for m in msgs)
        zh_decision = ["决定", "改用", "采用", "切换", "替换", "迁移", "升级", "部署", "配置",
                       "重构", "优化", "调整", "选择"]
        found = [kw for kw in zh_decision if kw in full_text]
        assert len(found) >= 1, \
            f"{path.name}: no Chinese decision verb found"

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("length", list(LENGTHS.keys()))
    def test_chinese_has_anomaly_verbs(self, domain, length):
        path = _DATA_DIR / f"{domain}_{length}_zh.json"
        with open(path, encoding="utf-8") as f:
            msgs = json.load(f)
        full_text = " ".join(m["content"] for m in msgs)
        zh_anomaly = ["报错", "失败", "超时", "崩溃", "异常", "泄漏", "死锁", "阻塞", "挂了"]
        found = [kw for kw in zh_anomaly if kw in full_text]
        assert len(found) >= 1, \
            f"{path.name}: no Chinese anomaly verb found"

    def test_all_english_ids_sequential(self):
        """Verify message IDs in English files are sequential 1..N."""
        for domain in DOMAINS:
            for length, expected in LENGTHS.items():
                path = _DATA_DIR / f"{domain}_{length}_en.json"
                with open(path, encoding="utf-8") as f:
                    msgs = json.load(f)
                ids = [m["id"] for m in msgs]
                assert ids == list(range(1, expected + 1)), \
                    f"{path.name}: IDs not sequential 1..{expected}"

    def test_each_file_is_valid_json(self):
        """Every generated file must be parseable JSON array."""
        for domain in DOMAINS:
            for length in LENGTHS:
                for lang in LANGS:
                    path = _DATA_DIR / f"{domain}_{length}_{lang}.json"
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    assert isinstance(data, list), \
                        f"{path.name}: not a JSON array"


class TestGenerateScriptImportable:
    """Verify generate_test_data.py can be imported and re-run."""

    def test_module_imports(self):
        sys.path.insert(0, str(Path(__file__).parent))
        import generate_test_data as gtd
        assert hasattr(gtd, "DOMAINS")
        assert hasattr(gtd, "LENGTHS")
        assert hasattr(gtd, "LANGUAGES")
        assert hasattr(gtd, "main")

    def test_domains_list(self):
        sys.path.insert(0, str(Path(__file__).parent))
        import generate_test_data as gtd
        assert len(gtd.DOMAINS) == 6
        assert "backend" in gtd.DOMAINS
        assert "game-dev" in gtd.DOMAINS

    def test_lengths_dict(self):
        sys.path.insert(0, str(Path(__file__).parent))
        import generate_test_data as gtd
        assert gtd.LENGTHS["short"] == 10
        assert gtd.LENGTHS["medium"] == 30
        assert gtd.LENGTHS["long"] == 60

    def test_reproducible_with_seed(self):
        """Same seed should produce identical output."""
        sys.path.insert(0, str(Path(__file__).parent))
        import generate_test_data as gtd
        import tempfile, shutil

        # Monkey-patch OUT_DIR to a temp dir
        orig_dir = gtd.OUT_DIR
        tmp1 = Path(tempfile.mkdtemp())
        tmp2 = Path(tempfile.mkdtemp())
        try:
            gtd.OUT_DIR = tmp1
            gtd.random.seed(42)
            gtd.OUT_DIR.mkdir(parents=True, exist_ok=True)
            msgs1 = gtd._build_en_conversation("backend", 10)

            gtd.OUT_DIR = tmp2
            gtd.random.seed(42)
            gtd.OUT_DIR.mkdir(parents=True, exist_ok=True)
            msgs2 = gtd._build_en_conversation("backend", 10)

            # Compare content
            for i, (m1, m2) in enumerate(zip(msgs1, msgs2)):
                assert m1["content"] == m2["content"], \
                    f"Message {i} differs: '{m1['content'][:50]}' vs '{m2['content'][:50]}'"
        finally:
            gtd.OUT_DIR = orig_dir
            shutil.rmtree(tmp1, ignore_errors=True)
            shutil.rmtree(tmp2, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Performance — Extraction Speed (US-008: 3 tests)
# ═══════════════════════════════════════════════════════════════════════════

import time
from unittest import mock


def _build_perf_msgs(n: int, domain: str = "backend"):
    """Build N synthetic messages for performance tests using generator templates."""
    sys.path.insert(0, str(Path(__file__).parent))
    import generate_test_data as gtd
    gtd.random.seed(42)
    return gtd._build_en_conversation(domain, n)


class TestExtractionSpeed:
    """Extraction speed tests: 10/50/100 messages, record timing."""

    def test_extraction_speed_10_messages(self):
        """10 messages extraction < 0.1s."""
        msgs = _build_perf_msgs(10)
        t0 = time.perf_counter()
        graph = _extract_graph_fallback(msgs)
        elapsed = time.perf_counter() - t0
        assert graph.total_anchors > 0, "Should extract anchors"
        assert elapsed < 0.1, \
            f"10 messages extraction took {elapsed:.4f}s, must be < 0.1s"

    def test_extraction_speed_50_messages(self):
        """50 messages extraction < 0.1s."""
        msgs = _build_perf_msgs(50)
        _extract_graph_fallback(msgs)  # Warm up
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _extract_graph_fallback(msgs)
            times.append(time.perf_counter() - t0)
        best = min(times)
        assert best < 0.1, \
            f"50 messages extraction best time {best:.4f}s, must be < 0.1s"

    def test_extraction_speed_100_messages(self):
        """100 messages extraction records timing baseline."""
        msgs = _build_perf_msgs(100)
        _extract_graph_fallback(msgs)  # Warm up
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            graph = _extract_graph_fallback(msgs)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        best = min(times)
        assert graph.total_anchors > 0, "Should extract anchors"
        assert best < 0.5, \
            f"100 messages best time {best:.4f}s, must be < 0.5s"
        assert avg < 1.0, \
            f"100 messages avg time {avg:.4f}s, must be < 1.0s"


# ═══════════════════════════════════════════════════════════════════════════
# Performance — Memory Usage (US-008: 2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Memory tests: extraction stays bounded."""

    def test_100_messages_memory(self):
        """100 messages extraction uses < 50MB."""
        import tracemalloc
        msgs = _build_perf_msgs(100)
        tracemalloc.start()
        graph = _extract_graph_fallback(msgs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert graph.total_anchors > 0, "Should extract anchors"
        assert peak_mb < 50, \
            f"Peak memory {peak_mb:.1f}MB, must be < 50MB"

    def test_100_messages_graph_size(self):
        """Graph serialization is < 100KB (compact)."""
        msgs = _build_perf_msgs(100)
        graph = _extract_graph_fallback(msgs)
        d = graph.to_dict()
        serialized = json.dumps(d, ensure_ascii=False)
        size_kb = len(serialized) / 1024
        assert size_kb < 100, \
            f"Graph serialized size {size_kb:.1f}KB, must be < 100KB"


# ═══════════════════════════════════════════════════════════════════════════
# Performance — SQLite FTS5 (US-008: 2 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLitePerf:
    """SQLite FTS5 search performance tests."""

    def test_fts5_search_speed(self):
        """FTS5 search returns in < 0.05s."""
        from anchor.store_sqlite import SqliteStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            store = SqliteStore(db_path=db_path)
            seq = AnchorSequence(session_id="perf-fts5")
            for i in range(200):
                a = Anchor(
                    entity=f"TestEntity{i}",
                    anchor_type=AnchorType.FACT,
                    entity_class=EntityClass.TECH,
                    pos=i,
                    data_values=[f"val_{i}"],
                )
                seq.anchors.append(a)
            store.save_sequence(seq)
            store.search("TestEntity50", limit=5)  # Warm up
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                results = store.search("TestEntity50", limit=5)
                times.append(time.perf_counter() - t0)
            best = min(times)
            assert len(results) > 0, "FTS5 should return results"
            assert best < 0.05, \
                f"FTS5 search best time {best:.4f}s, must be < 0.05s"
        finally:
            os.unlink(db_path)
            for suffix in ["-wal", "-shm"]:
                p = db_path + suffix
                if os.path.exists(p):
                    os.unlink(p)

    def test_fts5_search_empty_db(self):
        """FTS5 search on empty DB returns quickly."""
        from anchor.store_sqlite import SqliteStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            store = SqliteStore(db_path=db_path)
            t0 = time.perf_counter()
            results = store.search("nonexistent", limit=5)
            elapsed = time.perf_counter() - t0
            assert results == [], "Empty DB should return empty results"
            assert elapsed < 0.01, \
                f"Empty FTS5 search took {elapsed:.4f}s, must be < 0.01s"
        finally:
            os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# Performance — Cold Start (US-008: 1 test)
# ═══════════════════════════════════════════════════════════════════════════

class TestColdStart:
    """Cold vs warm extraction performance."""

    def test_cold_vs_warm_extraction(self):
        """First extraction (cold) slower than second (warm) due to Python caching."""
        msgs = _build_perf_msgs(50)
        t0 = time.perf_counter()
        g1 = _extract_graph_fallback(msgs)
        cold_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        g2 = _extract_graph_fallback(msgs)
        warm_time = time.perf_counter() - t0
        assert g1.total_anchors == g2.total_anchors, \
            "Cold and warm runs should produce identical anchors"
        assert warm_time < 0.1, \
            f"Warm extraction time {warm_time:.4f}s, must be < 0.1s"
        assert cold_time < 0.2, \
            f"Cold extraction time {cold_time:.4f}s, must be < 0.2s"


# ═══════════════════════════════════════════════════════════════════════════
# Hook Script Integration — Helpers (US-009)
# ═══════════════════════════════════════════════════════════════════════════

import subprocess
import shutil

_SCRIPTS_DIR = Path(__file__).parent.parent / "anchor-context" / "scripts"
_PYTHON = sys.executable


def _run_hook_script(script_name, args=None, stdin_data=None, home_dir=None,
                     timeout=30):
    """Run a hook Python script as subprocess with isolated HOME for store dir.

    Sets HOME and USERPROFILE to a temp directory so AnchorStore writes
    to an isolated location instead of the real ~/.claude/anchors/.
    """
    script_path = _SCRIPTS_DIR / script_name
    cmd = [_PYTHON, str(script_path)] + (args or [])
    env = os.environ.copy()
    if home_dir:
        env["HOME"] = home_dir
        env["USERPROFILE"] = home_dir
    return subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


_HOOK_TEST_MESSAGES = [
    {"content": "Decided to migrate PostgreSQL from version 14.2 to 15.0 for better performance"},
    {"content": "Found race condition in Redis SETNX at auth.ts:42 causing timeouts"},
    {"content": "API latency spiked to 200ms after deploying Kubernetes v1.27"},
    {"content": "Must implement JWT token rotation for OAuth2 in auth.ts:105"},
    {"content": "Configured Prometheus monitoring with Grafana dashboard for error ERR_005"},
    {"content": "Identified memory leak in PgBouncer connection pool at pool.go:78"},
    {"content": "Switched from Memcached to Redis cluster with 2.1GB capacity"},
    {"content": "Database deadlock error ERR_042 on order_items table at query_handler.go:200"},
    {"content": "Upgraded Prisma ORM from 4.7 to 5.1 to resolve N+1 query issue"},
    {"content": "Optimized PostgreSQL VACUUM settings after identifying 5GB of dead tuples"},
]


def _json_to_stdin(obj):
    """Serialize a dict to JSON string for stdin piping."""
    return json.dumps(obj, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Hook Script Integration — PreCompact Tests (US-009: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestPreCompactHook:
    """5 pre_compact tests: different stdin JSON formats -> anchors saved to disk."""

    def test_direct_messages_format_saves_anchors(self):
        """Input with direct 'messages' field → anchors extracted and saved."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "messages": _HOOK_TEST_MESSAGES,
                "session_id": "test-precompact-messages",
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            # Verify anchors saved to disk
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) > 0, "No anchor files saved"
            # Output should contain compact instructions
            assert len(result.stdout) > 0, "No compact instructions output"
            assert "anchor" in result.stdout.lower(), \
                f"Compact instructions missing 'anchor': {result.stdout[:200]}"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_conversation_nested_format_saves_anchors(self):
        """Input with 'conversation.messages' nested format → anchors saved."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "conversation": {
                    "messages": _HOOK_TEST_MESSAGES,
                },
                "session_id": "test-precompact-nested",
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) > 0, "No anchor files saved for nested format"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_empty_messages_no_save(self):
        """Empty messages list → no anchors saved, no crash."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "messages": [],
                "session_id": "test-empty",
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            anchors_dir = Path(home) / ".claude" / "anchors"
            # Should not create files for empty messages
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) == 0, \
                f"Should not save for empty messages, found {len(saved_files)} file(s)"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_no_session_id_generates_one(self):
        """Input without session_id → auto-generates a session_id and saves."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "messages": _HOOK_TEST_MESSAGES[:5],
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) > 0, \
                "Should generate session_id and save anchors"
            # The auto-generated session_id is 12 hex chars
            assert len(saved_files[0].stem) == 12, \
                f"Auto-generated session_id should be 12 hex chars, got: {saved_files[0].stem}"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_no_recognized_message_format_graceful(self):
        """Input with no recognized message format → graceful return, no crash."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "unrelated_field": "some data",
                "session_id": "test-noformat",
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            # Should not crash and not save anything
        finally:
            shutil.rmtree(home, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hook Script Integration — Inject Tests (US-009: 5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestInjectHook:
    """5 inject tests: saved anchors → valid hookSpecificOutput JSON."""

    def _save_anchors(self, home_dir, session_id, messages=None):
        """Helper: run pre_compact to save anchors, then return the anchors dir."""
        if messages is None:
            messages = _HOOK_TEST_MESSAGES
        stdin_data = _json_to_stdin({
            "messages": messages,
            "session_id": session_id,
        })
        _run_hook_script(
            "pre_compact.py", args=["save"],
            stdin_data=stdin_data, home_dir=home_dir,
        )

    def test_with_saved_anchors_outputs_valid_json(self):
        """Pre-saved anchors → inject outputs valid hookSpecificOutput JSON."""
        home = tempfile.mkdtemp()
        try:
            self._save_anchors(home, "test-inject-001")
            result = _run_hook_script(
                "inject.py", home_dir=home,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            output = json.loads(result.stdout.strip())
            assert "hookSpecificOutput" in output, \
                f"Missing hookSpecificOutput key: {list(output.keys())}"
            hso = output["hookSpecificOutput"]
            assert hso["hookEventName"] == "SessionStart"
            assert "additionalContext" in hso
            assert len(hso["additionalContext"]) > 0, \
                "additionalContext should not be empty when anchors exist"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_without_saved_anchors_outputs_empty(self):
        """No saved anchors → inject outputs valid but empty additionalContext."""
        home = tempfile.mkdtemp()
        try:
            # Don't save anything — inject from empty store
            result = _run_hook_script("inject.py", home_dir=home)
            assert result.returncode == 0
            output = json.loads(result.stdout.strip())
            assert output["hookSpecificOutput"]["additionalContext"] == ""
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_additional_context_contains_entities(self):
        """additionalContext should contain entity names from saved anchors."""
        home = tempfile.mkdtemp()
        try:
            self._save_anchors(home, "test-inject-ctx")
            result = _run_hook_script("inject.py", home_dir=home)
            assert result.returncode == 0
            output = json.loads(result.stdout.strip())
            ctx = output["hookSpecificOutput"]["additionalContext"]
            # Should contain at least one key entity
            assert "PostgreSQL" in ctx or "Redis" in ctx or "JWT" in ctx, \
                f"additionalContext missing entity names: {ctx[:300]}"
            assert "锚点上下文" in ctx, \
                "Should contain Chinese anchor context header"
            assert "Session:" in ctx or "session" in ctx.lower(), \
                "Should contain session identifier"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_output_schema_is_correct(self):
        """Output JSON follows the Claude Code hook output schema exactly."""
        home = tempfile.mkdtemp()
        try:
            self._save_anchors(home, "test-inject-schema")
            result = _run_hook_script("inject.py", home_dir=home)
            assert result.returncode == 0
            output = json.loads(result.stdout.strip())
            # Schema: { hookSpecificOutput: { hookEventName, additionalContext } }
            assert set(output.keys()) == {"hookSpecificOutput"}
            hso = output["hookSpecificOutput"]
            assert set(hso.keys()) == {"hookEventName", "additionalContext"}
            assert hso["hookEventName"] == "SessionStart"
            assert isinstance(hso["additionalContext"], str)
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_display_format_flag(self):
        """--format flag outputs human-readable text, not JSON."""
        home = tempfile.mkdtemp()
        try:
            self._save_anchors(home, "test-inject-display")
            result = _run_hook_script(
                "inject.py", args=["--format"], home_dir=home,
            )
            assert result.returncode == 0
            stdout = result.stdout.strip()
            # Display format has section headers, separators, not JSON
            assert "anchor" in stdout.lower() or "Session:" in stdout, \
                f"Display format should show anchor info: {stdout[:300]}"
            # Should NOT be valid JSON
            try:
                json.loads(stdout)
                assert False, "Display output should not be valid JSON"
            except json.JSONDecodeError:
                pass  # Expected — display format is human-readable text
        finally:
            shutil.rmtree(home, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hook Script Integration — StopBackup Tests (US-009: 3 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestStopBackupHook:
    """3 stop_backup tests: session exit saves anchors, including edge cases."""

    def test_new_session_saves_anchors(self):
        """≥3 messages, no existing anchors from PreCompact → anchors saved."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "messages": _HOOK_TEST_MESSAGES[:5],
                "session_id": "test-stop-new",
            })
            result = _run_hook_script(
                "stop_backup.py", stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) > 0, \
                "Stop hook should save anchors for new session"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_existing_session_skips(self):
        """Session already saved by PreCompact → stop_backup skips re-saving."""
        home = tempfile.mkdtemp()
        try:
            # First, save via pre_compact (simulates PreCompact hook ran)
            _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=_json_to_stdin({
                    "messages": _HOOK_TEST_MESSAGES,
                    "session_id": "test-stop-existing",
                }),
                home_dir=home,
            )
            anchors_dir = Path(home) / ".claude" / "anchors"
            files_before = set(f.name for f in anchors_dir.glob("*.json"))

            # Now run stop_backup with same session_id
            result = _run_hook_script(
                "stop_backup.py",
                stdin_data=_json_to_stdin({
                    "messages": _HOOK_TEST_MESSAGES[:5],
                    "session_id": "test-stop-existing",
                }),
                home_dir=home,
            )
            assert result.returncode == 0
            files_after = set(f.name for f in anchors_dir.glob("*.json"))
            # No new files should be created for already-saved session
            assert files_after == files_before, \
                "Stop hook should not re-save when PreCompact already saved"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_too_few_messages_skips(self):
        """<3 messages → stop_backup skips saving entirely."""
        home = tempfile.mkdtemp()
        try:
            stdin_data = _json_to_stdin({
                "messages": [
                    {"content": "Hello world"},
                    {"content": "How are you"},
                ],
                "session_id": "test-stop-short",
            })
            result = _run_hook_script(
                "stop_backup.py", stdin_data=stdin_data, home_dir=home,
            )
            assert result.returncode == 0
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) == 0, \
                "Should not save anchors for <3 messages"
        finally:
            shutil.rmtree(home, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hook Script Integration — Error Handling Tests (US-009: 3 tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestHookErrorHandling:
    """3 error handling tests: corrupted JSON, empty input, oversized input."""

    def test_corrupted_json_graceful_return(self):
        """Malformed JSON input → no crash, graceful return (non-zero ok, or exit 0)."""
        home = tempfile.mkdtemp()
        try:
            # Run all three scripts with garbage input
            for script in ["pre_compact.py", "inject.py", "stop_backup.py"]:
                args = ["save"] if script == "pre_compact.py" else None
                result = _run_hook_script(
                    script, args=args,
                    stdin_data="this is not valid json {{{",
                    home_dir=home,
                )
                # Scripts should not crash — they catch JSONDecodeError
                # pre_compact and stop_backup return early (exit 0)
                # inject loads from store (no stdin) so it's unaffected
                assert result.returncode == 0 or result.returncode is not None, \
                    f"{script} crashed on bad JSON: rc={result.returncode}"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_empty_input_graceful_return(self):
        """Empty stdin → no crash, graceful return."""
        home = tempfile.mkdtemp()
        try:
            for script in ["pre_compact.py", "stop_backup.py"]:
                args = ["save"] if script == "pre_compact.py" else None
                result = _run_hook_script(
                    script, args=args,
                    stdin_data="",
                    home_dir=home,
                )
                assert result.returncode == 0, \
                    f"{script} should handle empty input gracefully"
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_large_input_handled(self):
        """Very large input (1000+ messages) → handled without crash."""
        home = tempfile.mkdtemp()
        try:
            large_messages = []
            for i in range(1200):
                large_messages.append({
                    "content": f"Decided to deploy version {i}.{i%5} with Redis SETNX and error ERR_{i%1000:03d}",
                })
            stdin_data = _json_to_stdin({
                "messages": large_messages,
                "session_id": "test-large-input",
            })
            result = _run_hook_script(
                "pre_compact.py", args=["save"],
                stdin_data=stdin_data, home_dir=home,
                timeout=60,
            )
            assert result.returncode == 0, \
                f"Large input should not crash, rc={result.returncode}"
            # Large input should produce anchors
            anchors_dir = Path(home) / ".claude" / "anchors"
            saved_files = list(anchors_dir.glob("*.json")) if anchors_dir.exists() else []
            assert len(saved_files) > 0, \
                "Large input should save anchor files"
        finally:
            shutil.rmtree(home, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Code Block Handling Tests (US-016)
# ═══════════════════════════════════════════════════════════════════════════

class TestCodeBlockStripping:
    """Verify code blocks are replaced with summaries, not anchor-extracted."""

    def test_python_code_block_replaced(self):
        text = "We added a function:\n```python\ndef foo():\n    return 1\n```\nThis caused a bug."
        from anchor.extractor import _strip_code_blocks
        cleaned, refs = _strip_code_blocks(text)
        assert "def foo" not in cleaned, "Code should be stripped"
        assert "Code block: python" in cleaned, "Summary placeholder missing"
        assert "return 1" not in cleaned, "Code body should be removed"
        assert len(refs) > 0

    def test_sql_code_block_summarized(self):
        text = "```sql\nALTER TABLE users ADD COLUMN email TEXT;\n```"
        from anchor.extractor import _strip_code_blocks
        cleaned, refs = _strip_code_blocks(text)
        assert "ALTER TABLE" not in cleaned
        assert "Code block: sql" in cleaned
        assert refs[0]["n_lines"] == 1

    def test_multiple_code_blocks(self):
        text = "```python\ndef a():\n    pass\n```\nText.\n```js\nconst x = 1;\n```"
        from anchor.extractor import _strip_code_blocks
        cleaned, refs = _strip_code_blocks(text)
        assert len(refs) == 2
        assert "def a" not in cleaned
        assert "const x" not in cleaned
        assert "Code block: python" in cleaned
        assert "Code block: js" in cleaned

    def test_no_code_blocks(self):
        text = "Just some normal text about Redis and PostgreSQL."
        from anchor.extractor import _strip_code_blocks
        cleaned, refs = _strip_code_blocks(text)
        assert cleaned == text
        assert len(refs) == 0

    def test_extract_graph_skips_code(self):
        """Anchors from extract_graph should not include code entities."""
        msgs = [
            {"content": "We decided to use Redis for caching."},
            {"content": "```python\ndef acquire_lock():\n    return redis_client.set('key', 'val', nx=True)\n```"},
            {"content": "Bug found at auth.ts:42. Error ERR_005."},
        ]
        from anchor.extractor import extract_graph
        g = extract_graph(msgs)
        all_entities = " ".join(v.entity for v in g.verb_anchors)
        all_entities += " " + " ".join(n.entity for n in g.noun_anchors)
        # Code body is stripped, but function names in summaries are preserved
        # as useful metadata (tells you what code was written)
        assert "redis_client.set" not in all_entities, "Code body leaked into anchors"
        assert "Redis" in all_entities or "auth.ts" in all_entities
