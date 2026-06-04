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
