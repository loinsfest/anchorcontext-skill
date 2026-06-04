## Anchor Context v1.0.0

### Core
- Bidirectional verb-noun anchor graph with mutual nearest-neighbor links
- LLM judge mode (DeepSeek) -- replaces all hand-crafted rules with one API call
- Zero-cost fallback mode -- no API key needed
- 3 Claude Code hooks: PreCompact, SessionStart[compact], Stop
- ~93% compression on 30-message conversations

### Testing (604 tests, 0 failures)
- 315 test methods across 59 test classes
- Verb lexicon: 55 tests covering English/Chinese, past tense, compound verbs
- Entity extraction: 63 tests for DATA/TECH/TERM classification
- Bidirectional graph: 43 tests for link integrity, dedup, Top-N
- LLM judge: 29 tests for dual-mode (LLM + fallback)
- Compression benchmarks: 23 tests across 6 domains
- Reconstruction quality: 25 tests for query matching
- Performance: 8 tests (speed < 0.1s for 50msgs)
- Hook scripts: 16 tests for pre_compact/inject/stop_backup
- Auto-generated test data: 36 conversation files

### Install
```bash
git clone https://github.com/loinsfest/anchorcontext-skill.git
cd anchorcontext-skill && bash install.sh
```
