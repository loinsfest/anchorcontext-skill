# PRD: Ultra-Long Text Testing

## Introduction

项目此前仅在中长度对话（30-40 条消息，~900-1800 tokens）上验证通过。需要超长文本（100+ 条消息，10000+ tokens）验证压缩率、锚点质量、性能稳定性。

## Goals

- 生成 100/200/500 条消息的超长对话测试数据
- 验证压缩率在超长对话中保持 90%+
- 验证提取性能在 500 条消息下 < 2 秒
- 验证锚点质量：关键实体不丢失，噪声比 < 30%
- 验证 LLM 判断器在 200+ 候选时不超时

## User Stories

### US-011: Ultra-long conversation test data generation
**Description:** As a developer, I need 100/200/500-message conversations across 3 domains for stress testing.

**Acceptance Criteria:**
- [ ] Generate 100-msg backend conversation with decisions, anomalies, data values
- [ ] Generate 200-msg mixed conversation across backend/frontend/devops
- [ ] Generate 500-msg conversation spanning all 6 domains
- [ ] Each message has `id` and `content` fields
- [ ] Conversations saved to `tests/data/ultra-long/`
- [ ] Tests pass

### US-012: Compression ratio at scale
**Description:** As a developer, I need to verify compression stays 90%+ on ultra-long conversations.

**Acceptance Criteria:**
- [ ] 100-msg backend: compression >= 90%
- [ ] 200-msg mixed: compression >= 88%
- [ ] 500-msg full: compression >= 85%
- [ ] Anchor count scales sub-linearly: 500msgs should produce <= 250 anchors
- [ ] Tests pass

### US-013: Extraction performance at scale
**Description:** As a developer, I need to verify extraction stays fast on large inputs.

**Acceptance Criteria:**
- [ ] 100-msg extraction < 0.3s
- [ ] 200-msg extraction < 0.8s
- [ ] 500-msg extraction < 3.0s
- [ ] Memory usage at 500msgs < 100MB
- [ ] Tests pass

### US-014: Anchor quality at scale
**Description:** As a developer, I need to verify anchor quality doesn't degrade with conversation length.

**Acceptance Criteria:**
- [ ] All critical entities present (Redis, PostgreSQL, auth.ts, ERR_005, etc.)
- [ ] Noise ratio < 30% (common words, bare numbers without context)
- [ ] Verbat/noun balance: at least 20% verbs
- [ ] Link integrity: no dangling references
- [ ] Tests pass

### US-015: LLM judge at scale
**Description:** As a developer, I need to verify LLM judge handles 200+ candidates without timeout.

**Acceptance Criteria:**
- [ ] LLM judge processes 200 candidates within 30s (if API key available)
- [ ] Fallback handles 500 candidates correctly with quota
- [ ] LLM tags cover diverse domains (database, cache, auth, frontend, mobile)
- [ ] Tests pass
