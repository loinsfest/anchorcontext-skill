# PRD: Anchor Context — Production Stability Testing (500 Tests)

## Introduction

锚点上下文技能已通过 50 个基础单元测试，核心管道已验证。现在需要从多角度进行 500+ 次测试，目标是让它成为**生产级稳定项目**。测试涵盖：单元测试扩展、LLM 判断器双路径、Hook 集成、自动生成对话数据、压缩率基准、边界条件。

## Goals

- 从 50 个单元测试扩展到 200+ 个确定性测试
- 100 个自动化测试用例覆盖所有模块
- Ralph 自主循环执行 5 次迭代，每次运行完整测试套件 + 随机数据
- 自动生成随机对话数据（不同领域、长度、语言）
- 零回归——每次迭代后 200 个测试全绿
- 产出测试覆盖率 > 90%

## User Stories

### US-001: 扩展动词词表测试 (verbs.py)
**Description:** As a developer, I need to verify the 180+ verb lexicon correctly classifies all verb types across English past/present tense and Chinese.

**Acceptance Criteria:**
- [ ] 30 个 English verb tests: 覆盖 DECISION/DISCOVERY/ANOMALY/CONSTRAINT 四种类型，包含过去式 (decided, found, discovered, crashed)
- [ ] 10 个 Chinese verb tests: 决定/发现/报错/必须 等，含同义词变体
- [ ] 5 个 compound verb tests: "tracked down", "opted for", "narrowed down"
- [ ] 5 个 case-insensitive tests: "Decided", "FOUND", "Crashed"
- [ ] 5 个 unknown verb tests: 未收录动词应返回 FACT
- [ ] segment_text() 单次扫描性能: 1000 字文本 < 0.01s
- [ ] Typecheck/lint passes

### US-002: 扩展实体提取测试 (extractor.py)
**Description:** As a developer, I need to verify entity extraction correctly identifies DATA/TECH/TERM entities with proper classification and garbage filtering.

**Acceptance Criteria:**
- [ ] 15 个 DATA entity tests: 版本号(14.2, 3.10.1)、错误码(ERR_005)、数字+单位(200ms, 2.1GB)、行号(:42)
- [ ] 15 个 TECH entity tests: 文件名(auth.ts)、PascalCase(PostgreSQL, OAuth2)、camelCase、UPPER_CASE、域名(grafana.internal)
- [ ] 10 个 garbage filter tests: 句首大写词(Decided, Current)、裸数字(<10)、碎片词
- [ ] 5 个 Chinese entity tests: 分布式锁、跨Pod同步
- [ ] 5 个 mixed entity tests: 中英混合文本
- [ ] _is_proper_entity() 边界: Redis(5chars)、Decided(7chars)、Cannot(6chars)
- [ ] Typecheck/lint passes

### US-003: 扩展双向图提取测试 (extract_graph)
**Description:** As a developer, I need to verify extract_graph produces valid VerbAnchor+NounAnchor graphs with correct links, dedup, and Top-N selection.

**Acceptance Criteria:**
- [ ] 10 个 verb anchor tests: 每个动词正确链接到最近名词
- [ ] 10 个 noun anchor tests: 每个名词正确链接到最近动词，tags 不为空
- [ ] 8 个 link integrity tests: 无悬空引用，VerbAnchor.nearest_noun_id 指向存在的 NounAnchor
- [ ] 5 个 dedup tests: 重复实体/动词不会出现多次
- [ ] 5 个 Top-N tests: 30 条消息 → 12-20 锚点 (≤ max(8,30//2)=15)
- [ ] 5 个 empty/tiny input tests: 0条消息、1条消息、空白内容
- [ ] Typecheck/lint passes

### US-004: LLM 判断器测试 (judge.py)
**Description:** As a developer, I need to verify the LLM judge correctly selects significant anchors from candidates and generates meaningful tags, AND the fallback mode works when no API key is set.

**Acceptance Criteria:**
- [ ] 8 个 LLM mode tests: 给定 10 个候选，LLM 选择 6 个最重要的，标签非空
- [ ] 8 个 Fallback mode tests: 无 API key 时回退评分正常工作，动词+名词平衡
- [ ] 5 个 tag quality tests: LLM 标签覆盖数据库/缓存/认证/性能/监控等领域
- [ ] 5 个 edge case tests: 空候选列表、全噪声候选、仅有动词、仅有名词
- [ ] 3 个 API error tests: 网络超时、返回空 JSON、返回非法 JSON
- [ ] Typecheck/lint passes

### US-005: 压缩率基准测试 (benchmark)
**Description:** As a developer, I need to verify compression ratio stays above 85% across different conversation lengths and domains.

**Acceptance Criteria:**
- [ ] 5 个 backend domain tests: 30 条后端对话 → ≥85% 压缩
- [ ] 5 个 frontend domain tests: 40 条前端对话 → ≥85% 压缩
- [ ] 5 个 mixed domain tests: 50 条混合对话 → ≥80% 压缩
- [ ] 5 个 short conversation tests: 5-10 条对话 → 合理锚点数 (≤max(8, N//2))
- [ ] 3 个 extreme tests: 100 条消息 → ≥90% 压缩
- [ ] Typecheck/lint passes

### US-006: 重建质量测试 (reconstruction)
**Description:** As a developer, I need to verify that anchor-based reconstruction produces useful context windows that contain ground-truth information.

**Acceptance Criteria:**
- [ ] 10 个 query match tests: 用已知查询命中对应锚点，窗口包含期望关键词
- [ ] 5 个 tag-driven match tests: 语义标签使 "database"→"PostgreSQL" 匹配成功
- [ ] 5 个 link traversal tests: 从动词锚点→链接名词，或名词锚点→链接动词，窗口覆盖两个位置
- [ ] 5 个 negative tests: 不相关查询不应命中高置信度锚点
- [ ] Typecheck/lint passes

### US-007: 自动生成测试数据 (data generation)
**Description:** As a developer, I need a script that generates random conversations across 6 domains, 3 lengths, and 2 languages for comprehensive testing.

**Acceptance Criteria:**
- [ ] 脚本 `tests/generate_test_data.py` 可运行
- [ ] 6 个领域: backend, frontend, devops, data-science, mobile, game-dev
- [ ] 3 种长度: short (10msgs), medium (30msgs), long (60msgs)
- [ ] 2 种语言: English, Chinese
- [ ] 每个对话包含: 决策动词、异常、数据值(版本/行号/错误码)
- [ ] 生成 36 个对话文件 (6×3×2) 保存到 `tests/data/`
- [ ] Typecheck/lint passes

### US-008: 性能基准测试 (performance)
**Description:** As a developer, I need to verify extraction stays fast (< 0.1s for 50 messages) and memory usage is bounded.

**Acceptance Criteria:**
- [ ] 3 个 extraction speed tests: 10/50/100 条消息，记录时间
- [ ] 2 个 memory tests: 100 条消息内存 < 50MB
- [ ] 2 个 SQLite perf tests: FTS5 搜索 < 0.05s
- [ ] 1 个 cold start test: 首次提取（无缓存）vs 后续提取
- [ ] Typecheck/lint passes

### US-009: Hook 脚本测试 (hooks)
**Description:** As a developer, I need to verify all 3 hook scripts (pre_compact, inject, stop_backup) work correctly with stdin/stdout.

**Acceptance Criteria:**
- [ ] 5 个 pre_compact tests: 不同格式的 stdin JSON → anchors saved
- [ ] 5 个 inject tests: 从保存的锚点生成有效的 hookSpecificOutput JSON
- [ ] 3 个 stop_backup tests: 会话退出时保存锚点（含边界情况）
- [ ] 3 个 error handling tests: 损坏的 JSON、空输入、超大输入
- [ ] Typecheck/lint passes

### US-010: Ralph 自主循环执行 (autonomous loop)
**Description:** As a developer, I need Ralph to autonomously run the full test suite 5 times with random conversation data, catching flaky tests and regressions.

**Acceptance Criteria:**
- [ ] 创建 `prd.json` 包含以上 10 个 user stories
- [ ] Ralph 循环 5 次迭代: 每次启动新鲜 Claude Code 实例
- [ ] 每次迭代: 运行全部 200+ 测试 → 记录通过/失败 → 修失败 → 提交
- [ ] 5 次迭代后: 所有测试稳定通过，零 flaky
- [ ] progress.txt 记录每轮通过率和学习

## Functional Requirements

- FR-1: 测试总数 ≥ 200 个（确定性单元+集成测试）
- FR-2: 配合自动生成数据 + 参数化，总计执行 ≥ 500 次断言
- FR-3: 测试覆盖率: models.py ≥ 95%, extractor.py ≥ 90%, judge.py ≥ 85%, verbs.py ≥ 90%
- FR-4: 每个测试在 2 秒内完成
- FR-5: 零外部依赖——测试不需要 API key、网络、或 GPU
- FR-6: `python -m pytest tests/ -v` 一键运行全部测试
- FR-7: 所有测试在 Windows (Git Bash) 和 Linux 下通过

## Non-Goals

- 不需要 CLI 界面或 Web 仪表盘
- 不需要 CI/CD pipeline 配置（可在 GitHub Actions 中后续添加）
- 不测试 Claude Code 内部行为（compaction、hook 触发——这些属于 Claude Code 本身）
- 不进行 LLM API 调用的负载测试（成本考虑）

## Technical Considerations

- 测试框架: pytest 9.x（已安装）
- 自动生成对话: 使用模板 + 随机组合，不依赖外部 API
- LLM judge 测试: 默认用 fallback 模式，LLM 模式仅在设置 `ANCHOR_JUDGE_API_KEY` 时启用
- 性能测试: 使用 `time` 或 `pytest-benchmark`
- 内存测试: 使用 `tracemalloc` 或 `psutil`

## Success Metrics

- 200+ 测试通过 (pytest): 退出码 0
- 压缩率基准: 后端 ≥85%, 前端 ≥85%, 混合 ≥80%
- 重建评分: 平均 ≥5/10（当前基线 5.7）
- 提取速度: 50 条消息 < 0.1s
- Ralph 5 次迭代: 全部通过，无 flaky 测试
