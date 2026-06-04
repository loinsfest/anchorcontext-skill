"""Generate test conversation data across 6 domains x 3 lengths x 2 languages.

Output: tests/data/{domain}_{length}_{lang}.json (36 files total)
Each file is a JSON array of {"id": N, "content": "..."} message dicts.

Usage: python tests/generate_test_data.py
"""

import json
import os
import random
import sys
from pathlib import Path

random.seed(42)  # Reproducible generation

# ── Output directory ─────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "data"

# ── Domain definitions ────────────────────────────────────────────────────
DOMAINS = ["backend", "frontend", "devops", "data-science", "mobile", "game-dev"]
LENGTHS = {"short": 10, "medium": 30, "long": 60}
LANGUAGES = {"en": "English", "zh": "Chinese"}

# ── Shared data values pool ───────────────────────────────────────────────
VERSIONS = ["14.2", "3.10.1", "2.8.0", "1.5.3", "4.0.0", "7.1.2", "2.3.11", "5.2.0"]
ERROR_CODES = ["ERR_001", "ERR_005", "ERR_042", "ERR_103", "ERR_500", "WARN_007", "DB_001", "AUTH_003"]
NUMBERS_WITH_UNITS = ["200ms", "80ms", "2.1GB", "180MB", "500RPS", "100req/s", "45ms", "88ms", "150ms", "3.2GB", "5s", "10min"]
LINE_NUMBERS = [":42", ":103", ":256", ":512", ":1024", ":37", ":88", ":15"]

# ── English domain templates ──────────────────────────────────────────────

EN_TEMPLATES = {
    "backend": {
        "entities": ["Redis", "PostgreSQL", "PgBouncer", "JWT", "OAuth2", "TOTP", "Kubernetes", "Docker", "API", "GraphQL", "REST", "WebSocket", "gRPC", "Prometheus", "Grafana", "Nginx", "Envoy", "Kafka", "RabbitMQ", "ETCD"],
        "filenames": ["auth.ts", "middleware.py", "router.go", "handler.rs", "service.java", "model.rb", "schema.sql", "config.yaml"],
        "verbs_decision": ["decided", "chose", "switched", "opted for", "adopted", "migrated", "replaced", "configured", "deployed", "upgraded"],
        "verbs_discovery": ["found", "discovered", "identified", "traced", "located", "diagnosed", "pinpointed", "detected"],
        "verbs_anomaly": ["error", "crashed", "timeout", "failed", "broken", "corrupted", "missing", "overflow", "leak", "deadlock"],
    },
    "frontend": {
        "entities": ["React", "Vite", "Webpack", "Tailwind", "Zustand", "Redux", "Prisma", "Radix", "Storybook", "Playwright", "Vitest", "Clerk", "Next.js", "tRPC", "TanStack", "Figma", "Chromatic", "Lighthouse", "WCAG", "CSP"],
        "filenames": ["App.tsx", "Button.tsx", "useAuth.ts", "store.ts", "tailwind.config.js", "vite.config.ts", "index.html", "api.ts"],
        "verbs_decision": ["decided", "chose", "switched", "opted for", "replaced", "adopted", "refactored", "configured", "migrated"],
        "verbs_discovery": ["found", "discovered", "identified", "noticed", "detected", "observed", "traced"],
        "verbs_anomaly": ["error", "broken", "missing", "crash", "leak", "timeout", "degraded", "fail"],
    },
    "devops": {
        "entities": ["Kubernetes", "Docker", "Terraform", "Helm", "ArgoCD", "Prometheus", "Grafana", "Datadog", "PagerDuty", "AWS", "Cloudflare", "Vercel", "GitHub Actions", "Jenkins", "Ansible", "Istio", "Linkerd", "Fluentd", "Elasticsearch", "Kibana"],
        "filenames": ["Dockerfile", "deploy.yaml", "helmfile.yaml", "main.tf", "prometheus.yml", "alerts.yml", "pipeline.yml", "docker-compose.yaml"],
        "verbs_decision": ["decided", "chose", "switched", "migrated", "deployed", "configured", "upgraded", "replaced", "adopted"],
        "verbs_discovery": ["found", "discovered", "identified", "detected", "traced", "diagnosed", "located", "pinpointed"],
        "verbs_anomaly": ["error", "crashed", "timeout", "OOM", "panic", "degraded", "fail", "leak", "bottleneck"],
    },
    "data-science": {
        "entities": ["PyTorch", "TensorFlow", "Jupyter", "Pandas", "NumPy", "scikit-learn", "XGBoost", "MLflow", "Kubeflow", "Spark", "Databricks", "Airflow", "dbt", "Snowflake", "BigQuery", "Feast", "TFDV", "SHAP", "Optuna", "Weights & Biases"],
        "filenames": ["train.py", "model.pkl", "pipeline.py", "features.yaml", "notebook.ipynb", "eval.py", "config.yaml", "dataset.csv"],
        "verbs_decision": ["decided", "chose", "switched", "adopted", "tuned", "selected", "configured", "optimized", "replaced"],
        "verbs_discovery": ["found", "discovered", "identified", "observed", "detected", "noticed", "realized", "diagnosed"],
        "verbs_anomaly": ["error", "fail", "degraded", "bias", "drift", "overfit", "leak", "missing", "corrupted"],
    },
    "mobile": {
        "entities": ["React Native", "SwiftUI", "Kotlin", "Flutter", "Expo", "Fastlane", "Xcode", "Android Studio", "Firebase", "Crashlytics", "AppCenter", "CodePush", "Detox", "Maestro", "Appium", "CocoaPods", "Gradle", "ProGuard"],
        "filenames": ["App.tsx", "Podfile", "build.gradle", "Info.plist", "AndroidManifest.xml", "fastlane.rb", "app.json", "MainActivity.kt"],
        "verbs_decision": ["decided", "chose", "switched", "adopted", "upgraded", "replaced", "configured", "migrated", "deployed"],
        "verbs_discovery": ["found", "discovered", "identified", "traced", "detected", "located", "diagnosed", "noticed"],
        "verbs_anomaly": ["crash", "error", "timeout", "leak", "fail", "broken", "missing", "degraded", "hang"],
    },
    "game-dev": {
        "entities": ["Unity", "Unreal Engine", "Godot", "Photon", "FMOD", "Wwise", "OpenGL", "Vulkan", "DirectX", "Blender", "Maya", "Substance", "Houdini", "Spine", "Box2D", "Bullet", "Havok", "EOS", "PlayFab", "Steamworks"],
        "filenames": ["Player.cpp", "PhysicsWorld.h", "renderer.cpp", "shader.glsl", "level.json", "animator.py", "network.go", "build.ps1"],
        "verbs_decision": ["decided", "chose", "switched", "replaced", "adopted", "optimized", "configured", "refactored", "upgraded"],
        "verbs_discovery": ["found", "discovered", "identified", "traced", "detected", "located", "pinpointed", "diagnosed"],
        "verbs_anomaly": ["crash", "error", "hang", "leak", "broken", "corrupted", "missing", "timeout", "fail"],
    },
}

# ── Chinese domain templates ──────────────────────────────────────────────

ZH_TEMPLATES = {
    "backend": {
        "entities": ["Redis", "PostgreSQL", "分布式锁", "消息队列", "微服务", "API网关", "数据库连接池", "缓存穿透", "JWT认证", "数据库"],
        "filenames": ["auth.py", "middleware.go", "router.rs", "schema.sql", "config.yaml", "service.java", "handler.ts"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "替换", "迁移", "升级", "部署", "配置"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "排查", "识别", "检测到", "注意到"],
        "verbs_anomaly": ["报错", "失败", "超时", "崩溃", "挂了", "阻塞", "泄漏", "异常", "死锁"],
    },
    "frontend": {
        "entities": ["React", "Vue", "组件库", "状态管理", "构建工具", "CSS方案", "Next.js", "服务端渲染", "前端性能", "Webpack"],
        "filenames": ["App.tsx", "Button.vue", "store.ts", "useAuth.ts", "vite.config.ts", "tailwind.config.js", "api.ts"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "替换", "重构", "优化", "配置", "迁移"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "排查", "识别", "注意到", "检测到"],
        "verbs_anomaly": ["报错", "失败", "崩溃", "有问题", "不生效", "无效", "丢失", "异常"],
    },
    "devops": {
        "entities": ["Kubernetes", "Docker", "CI/CD", "监控系统", "日志平台", "自动化部署", "容器编排", "服务网格", "配置管理"],
        "filenames": ["Dockerfile", "deploy.yaml", "pipeline.yml", "prometheus.yml", "main.tf", "helmfile.yaml"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "替换", "迁移", "升级", "部署", "配置"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "排查", "识别", "检测到", "追踪"],
        "verbs_anomaly": ["报错", "失败", "超时", "崩溃", "OOM", "异常", "阻塞", "打满"],
    },
    "data-science": {
        "entities": ["模型训练", "特征工程", "数据管道", "A/B测试", "离线评估", "在线推理", "模型部署", "数据质量", "特征存储"],
        "filenames": ["train.py", "pipeline.py", "model.pkl", "features.yaml", "eval.py", "config.yaml"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "选择", "优化", "调整", "替换", "配置"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "分析出", "识别", "检测到", "观察到"],
        "verbs_anomaly": ["报错", "失败", "异常", "偏差", "过拟合", "泄漏", "丢失", "下降"],
    },
    "mobile": {
        "entities": ["React Native", "Flutter", "原生模块", "热更新", "应用性能", "崩溃监控", "应用发布", "推送服务", "离线缓存"],
        "filenames": ["App.tsx", "Podfile", "build.gradle", "AndroidManifest.xml", "Info.plist", "fastlane.rb"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "替换", "升级", "配置", "迁移", "部署"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "排查", "识别", "检测到", "注意到"],
        "verbs_anomaly": ["崩溃", "报错", "失败", "超时", "卡住", "泄漏", "异常", "无效"],
    },
    "game-dev": {
        "entities": ["Unity", "Unreal", "游戏引擎", "物理系统", "渲染管线", "网络同步", "资源管理", "动画系统", "AI行为树"],
        "filenames": ["Player.cpp", "Renderer.cpp", "shader.glsl", "level.json", "PhysicsWorld.h", "NetworkManager.cs"],
        "verbs_decision": ["决定", "改用", "采用", "切换", "替换", "优化", "重构", "升级", "配置"],
        "verbs_discovery": ["发现", "定位", "确认", "找到", "排查", "追踪", "识别", "检测到"],
        "verbs_anomaly": ["崩溃", "报错", "失败", "卡住", "泄漏", "异常", "有问题", "丢帧"],
    },
}

# Chinese conversation structure phrases
ZH_PHRASES = {
    "architecture": [
        "我们{verb_decision}使用{entity}作为{context}。当前系统存在{problem}问题。",
        "架构评审后，团队{verb_decision}{entity}来处理{context}，{data_value}版本已通过测试。",
        "为了支持{context}，{verb_decision}将{old_entity}迁移到{entity}。",
    ],
    "bug": [
        "在{filename}中{verb_discovery}了一个{verb_anomaly}问题，错误码{error_code}，出现在第{line_num}行。",
        "排查发现{filename}第{line_num}行{entity}连接{verb_anomaly}，响应时间达到{number_unit}，超过SLA阈值。",
        "根因分析：{filename}中{entity}的{verb_anomaly}导致{impact}。已{verb_decision}修复方案。",
    ],
    "fix": [
        "{verb_decision}将{entity}的{param}设置为{value}。测试显示延迟从{number_unit}降到{number_unit}。",
        "修复方案：{verb_decision}添加{entity}限制。部署后错误率从{value}%降至{value}%。",
        "{verb_decision}使用{entity}替代{old_entity}，内存占用从{number_unit}减少到{number_unit}。",
    ],
    "monitoring": [
        "添加了{entity}监控指标，告警阈值设为{number_unit}。面板地址：{entity_domain}。",
        "负载测试结果：{value}RPS持续{value}分钟，p95延迟{number_unit}，零报错。",
        "{verb_discovery}{entity}的{verb_anomaly}异常：{param}从{number_unit}增长到{number_unit}。",
    ],
    "deploy": [
        "已部署{entity}到生产环境{version}版本。灰度发布：{value}%流量持续{value}分钟。关键代码在{line_ref}。",
        "{entity}上线后{verb_anomaly}率下降{value}%，平均响应时间{number_unit}。",
        "代码覆盖率{value}%，{value}个集成测试全部通过。{verb_decision}合并到主分支。",
    ],
}

# ── English conversation templates ─────────────────────────────────────────

def _pick(items, n=1):
    if n == 1:
        return random.choice(items)
    return random.sample(items, min(n, len(items)))


def _make_en_message(msg_id, domain, msg_template):
    """Build an English message with variable substitution."""
    dt = EN_TEMPLATES[domain]
    ents = _pick(dt["entities"], 3)
    fnames = _pick(dt["filenames"], 2)
    ver = _pick(VERSIONS)
    ec = _pick(ERROR_CODES)
    nu = _pick(NUMBERS_WITH_UNITS)
    ln = _pick(LINE_NUMBERS)
    vd = _pick(dt["verbs_decision"], 3)
    vdis = _pick(dt["verbs_discovery"], 2)
    va = _pick(dt["verbs_anomaly"], 2)
    num = str(random.randint(10, 99))
    pct = str(random.randint(10, 99))

    return {"id": msg_id, "content": msg_template.format(
        entity=ents[0],
        old_entity=ents[1] if len(ents) > 1 else ents[0],
        entity2=ents[-1],
        filename=fnames[0],
        fname2=fnames[1] if len(fnames) > 1 else fnames[0],
        version=ver,
        error_code=ec,
        number_unit=nu,
        line_num=ln,
        verb_decision=vd[0],
        vd2=vd[1] if len(vd) > 1 else vd[0],
        verb_discovery=vdis[0],
        verb_anomaly=va[0],
        va2=va[1] if len(va) > 1 else va[0],
        num=num, pct=pct,
        verb_decision2=vd[1] if len(vd) > 1 else vd[0],
    )}


def _make_zh_message(msg_id, domain, phase, zh_tmpl, context_words):
    """Build a Chinese message with variable substitution."""
    ents = _pick(ZH_TEMPLATES[domain]["entities"], 3)
    fnames = _pick(ZH_TEMPLATES[domain]["filenames"], 2)
    ver = _pick(VERSIONS)
    ec = _pick(ERROR_CODES)
    nu = _pick(NUMBERS_WITH_UNITS)
    ln = _pick(LINE_NUMBERS)
    vd = _pick(ZH_TEMPLATES[domain]["verbs_decision"], 2)
    vdis = _pick(ZH_TEMPLATES[domain]["verbs_discovery"], 2)
    va = _pick(ZH_TEMPLATES[domain]["verbs_anomaly"], 2)
    num = str(random.randint(10, 99))
    pct = str(random.randint(10, 99))
    rint = str(random.randint(100, 999))
    val = str(random.randint(10, 99))

    # Context-dependent placeholders
    replacements = {
        "entity": ents[0],
        "old_entity": ents[1] if len(ents) > 1 else ents[0],
        "entity_domain": f"{ents[0].lower() if ents[0].isascii() else ents[0]}.internal",
        "filename": fnames[0],
        "fname2": fnames[1] if len(fnames) > 1 else fnames[0],
        "version": ver,
        "error_code": ec,
        "number_unit": nu,
        "line_num": ln.replace(":", ""),
        "line_ref": ln,
        "verb_decision": vd[0],
        "vd2": vd[1] if len(vd) > 1 else vd[0],
        "verb_discovery": vdis[0],
        "verb_anomaly": va[0],
        "va2": va[1] if len(va) > 1 else va[0],
        "context": _pick(context_words),
        "problem": _pick(["性能瓶颈", "扩展性不足", "安全漏洞", "数据不一致", "高延迟"]),
        "impact": _pick(["用户体验下降", "数据丢失", "服务不可用", "告警风暴", "客服工单激增"]),
        "param": _pick(["超时时间", "连接池大小", "缓存大小", "并发数", "批次大小"]),
        "value": rint,
        "num": num,
        "pct": pct,
        "val": val,
    }
    content = zh_tmpl.format(**replacements)
    return {"id": msg_id, "content": content}


# ── English conversation builders ──────────────────────────────────────────

EN_PHASES = {
    "architecture": [
        "We need to build a {entity}-based system for {entity2}. The current {old_entity} setup cannot handle our scale.",
        "I propose using {entity}. {verb_decision} to adopt {entity} {version} as the primary {entity2} layer.",
        "Architecture review: {verb_decision} to use {entity} with {entity2}. Estimated {num}% improvement in throughput.",
        "The team {verb_decision} to replace {old_entity} with {entity} after benchmarking both at {number_unit} latency.",
        "{verb_decision} to split the monolith — {entity} will handle {entity2} independently via {filename}.",
        "Connection pool: {verb_decision} to configure {entity} with pool size {num}, timeout {number_unit}.",
    ],
    "bug": [
        "{verb_discovery} a critical {verb_anomaly} in {filename} at line {line_num}. Error code: {error_code}.",
        "Root cause: {entity} {verb_anomaly} when concurrent requests exceed {num}. Causes intermittent {va2}.",
        "{verb_discovery} that {entity} is {verb_anomaly} under load — response times spike from {number_unit} to {number_unit}.",
        "This {verb_anomaly} explains the {error_code} errors in production. {entity} was {va2} across all pods.",
        "Traced the {verb_anomaly} to {entity} version {version} — a regression introduced in the last release.",
        "Also {verb_discovery} a memory {verb_anomaly}: {entity} heap grew from {number_unit} to {number_unit} in {num} hours.",
    ],
    "fix": [
        "Fix: {verb_decision} to acquire {entity} lock before {entity2} validation. Timeout: {number_unit}.",
        "{verb_decision} to add rate limiting — max {num} requests per minute via {entity}. Using {entity2} pattern.",
        "Deployed fix to staging. {entity} response time dropped from {number_unit} to {number_unit} after the change.",
        "{verb_decision} to increase {entity} pool to {num}, with connection timeout of {number_unit}.",
        "Added {entity} guard: {verb_decision} to reject requests when {entity2} exceeds {num}% capacity.",
        "{verb_decision} to switch from {old_entity} to {entity}. Memory usage dropped {pct}% after migration.",
    ],
    "monitoring": [
        "Added {entity} metrics: throughput, latency p95, error rate. Dashboard at {entity}.internal.",
        "Load test results: {num} RPS sustained, p50={number_unit}, p95={number_unit}, p99={number_unit}. Zero {verb_anomaly}s.",
        "{verb_discovery} that {entity} CPU spikes to {pct}% every {num} minutes — correlated with {entity2} batch jobs.",
        "Alert threshold set: {entity} p95 latency > {number_unit} triggers PagerDuty. Oncall dashboard at {entity}.internal/d/{entity2}.",
        "{entity} coverage now {pct}% — {num} integration tests, {num} unit tests. Main gap is {entity2} edge cases.",
        "Post-deploy monitoring shows {verb_anomaly} rate dropped from {pct}% to {num}%. SLA compliance restored.",
    ],
    "deploy": [
        "Deployed {entity} {version} to production. Canary: {num}% for {num} min, then full rollout.",
        "Rollback plan: if {verb_anomaly} rate exceeds {num}%, revert to {entity} {version} via feature flag.",
        "This sprint: {verb_decision} to implement {entity} support in {filename}. Estimated {num} story points.",
        "Code coverage at {pct}% — all {num} tests pass. {verb_decision} to ship {entity} {version}.",
        "Next milestone: integrate {entity} with {entity2}. Must maintain backward compatibility with {version} clients.",
        "CI pipeline now runs {num} tests across {entity} and {entity2}. Build time: {number_unit}.",
    ],
}


def _build_en_conversation(domain, n_msgs):
    """Build an English domain conversation with N messages."""
    msgs = []
    phase_names = ["architecture", "bug", "fix", "monitoring", "deploy"]
    tmpl = EN_TEMPLATES[domain]
    for i in range(n_msgs):
        phase = phase_names[i % len(phase_names)]
        templates = EN_PHASES[phase]
        chosen = templates[i % len(templates)]
        try:
            msg = _make_en_message(i + 1, domain, chosen)
        except (KeyError, IndexError):
            msg = {"id": i + 1, "content": f"Continuing work on {EN_TEMPLATES[domain]['entities'][0]} integration."}
        msgs.append(msg)
    return msgs


# ── Chinese conversation builders ───────────────────────────────────────────

def _build_zh_conversation(domain, n_msgs):
    """Build a Chinese domain conversation with N messages."""
    msgs = []
    context_words = {
        "backend": ["用户认证", "分布式缓存", "数据库优化", "API网关", "消息队列", "微服务通信"],
        "frontend": ["页面渲染", "组件开发", "状态管理", "构建优化", "样式方案", "性能调优"],
        "devops": ["持续部署", "监控告警", "日志收集", "容器编排", "配置管理", "灾备切换"],
        "data-science": ["模型训练", "特征工程", "数据清洗", "在线推理", "离线评估", "A/B测试"],
        "mobile": ["页面渲染", "推送通知", "离线存储", "性能优化", "权限管理", "热更新"],
        "game-dev": ["渲染优化", "物理模拟", "网络同步", "资源加载", "动画系统", "AI行为"],
    }
    ctx = context_words.get(domain, context_words["backend"])

    # Build messages from Chinese phrase templates
    phase_keys = ["architecture", "bug", "fix", "monitoring", "deploy"]
    for i in range(n_msgs):
        phase_key = phase_keys[i % len(phase_keys)]
        templates = ZH_PHRASES[phase_key]
        chosen = templates[i % len(templates)]
        try:
            msg = _make_zh_message(i + 1, domain, phase_key, chosen, ctx)
        except (KeyError, IndexError):
            tmpl = ZH_TEMPLATES[domain]
            msg = {"id": i + 1, "content": f"继续推进{tmpl['entities'][0]}集成工作，完善{tmpl['filenames'][0]}模块。"}
        msgs.append(msg)
    return msgs


# ── File writer ─────────────────────────────────────────────────────────────

def write_conversation(domain, length, lang, msgs):
    """Write a conversation to tests/data/{domain}_{length}_{lang}.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{domain}_{length}_{lang}.json"
    filepath = OUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)
    return filepath


# ── Validation ──────────────────────────────────────────────────────────────

def validate_conversation(msgs, domain, length, lang):
    """Validate a conversation meets minimum requirements."""
    full_text = " ".join(m["content"] for m in msgs)
    issues = []

    if len(msgs) != LENGTHS[length]:
        issues.append(f"Expected {LENGTHS[length]} messages, got {len(msgs)}")

    if lang == "en":
        required_verb_types = ["decided", "found", "error", "must"]
    else:
        required_verb_types = ["决定", "发现", "报错", "必须"]

    has_version = any(v in full_text for v in VERSIONS)
    has_error = any(e in full_text for e in ERROR_CODES)
    has_number = any(n in full_text for n in NUMBERS_WITH_UNITS)
    has_line = any(l in full_text for l in LINE_NUMBERS)

    if not has_version:
        issues.append("No version number found")
    if not has_error:
        issues.append("No error code found")
    if not has_number:
        issues.append("No number-with-unit found")
    if not has_line:
        issues.append("No line number found")

    return issues


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total = 0
    failed = []

    print(f"Generating test conversation data...")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print()

    for domain in DOMAINS:
        for length_key, n_msgs in LENGTHS.items():
            # English
            en_msgs = _build_en_conversation(domain, n_msgs)
            en_path = write_conversation(domain, length_key, "en", en_msgs)
            issues = validate_conversation(en_msgs, domain, length_key, "en")
            if issues:
                failed.append((str(en_path), issues))
            else:
                total += 1

            # Chinese
            zh_msgs = _build_zh_conversation(domain, n_msgs)
            zh_path = write_conversation(domain, length_key, "zh", zh_msgs)
            issues = validate_conversation(zh_msgs, domain, length_key, "zh")
            if issues:
                failed.append((str(zh_path), issues))
            else:
                total += 1

    for path, issues in failed:
        print(f"  WARNING: {path} — {', '.join(issues)}")

    print(f"\nGenerated {total}/{36} files successfully ({(total/36)*100:.0f}%).")
    if failed:
        print(f"{len(failed)} files had validation warnings.")
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
