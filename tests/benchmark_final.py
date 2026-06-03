# -*- coding: utf-8 -*-
"""Final benchmark: all compression methods including LLM judge."""
import sys, os, re, random
random.seed(42)

sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~/.claude/skills/anchor-context/scripts'))
# Set your DeepSeek API key to enable LLM judge mode
# os.environ['DEEPSEEK_API_KEY'] = 'your-key-here'

from anchor.extractor import extract_graph
import test_long_conversation as t1
import test_long_conversation2 as t2


def score(text, gt):
    mh = sum(1 for t in gt['must_contain'] if t.lower() in text.lower())
    sh = sum(1 for t in gt['should_contain'] if t.lower() in text.lower())
    return min(10, (mh/len(gt['must_contain']))*10 + (sh/max(1,len(gt['should_contain'])))*2)


def slide_text(msgs, pct):
    n = max(1, int(len(msgs)*pct))
    return ' '.join(m['content'] for m in msgs[-n:])


def comp_text(msgs):
    first = [m['content'] for m in msgs[:2]]
    last = [m['content'] for m in msgs[-5:]]
    sents = []
    for m in msgs[2:-5]:
        for s in re.split(r'[.!?]\s+', m['content']):
            if len(s) > 30:
                n = len(re.findall(r'[A-Z][a-z]{2,}|\d+\.?\d*', s))
                sents.append((n, s.strip()))
    sents.sort(reverse=True)
    key = [s for _, s in sents[:5]]
    return ' '.join(first + key + last)


def extr_text(msgs, q):
    sents = []
    for m in msgs:
        for s in re.split(r'[.!?]\s+', m['content']):
            if len(s) > 15:
                sents.append(s.strip())
    qw = set(q.lower().split())
    return ' '.join([s for s in sents if any(w in s.lower() for w in qw)][:8])


def anchor_text(msgs, q, use_llm=False):
    g = extract_graph(msgs)
    ft = ' '.join(m['content'] for m in msgs)
    best = (0, None, None)
    ql = q.lower()
    for v in g.verb_anchors:
        st = v.entity + ' ' + ' '.join(v.data_hints)
        s = sum(1 for w in ql.split() if w in st.lower())
        if s > best[0]:
            best = (s, v, 'v')
    for n in g.noun_anchors:
        st = n.entity + ' ' + ' '.join(n.tags) + ' ' + ' '.join(n.data_values)
        s = sum(1 for w in ql.split() if w in st.lower())
        if s > best[0]:
            best = (s, n, 'n')
    if best[1]:
        positions = [best[1].pos]
        if best[2] == 'v' and best[1].nearest_noun_id:
            p = g.find_noun(best[1].nearest_noun_id)
            if p: positions.append(p.pos)
        elif best[2] == 'n' and best[1].nearest_verb_id:
            p = g.find_verb(best[1].nearest_verb_id)
            if p: positions.append(p.pos)
        ws, we = max(0, min(positions) - 100), min(len(ft), max(positions) + 100)
        return ft[ws:we]
    return ""


def anchor_tok(msgs):
    g = extract_graph(msgs)
    return g.total_chars // 4


def slide_tok(msgs, pct):
    n = max(1, int(len(msgs) * pct))
    return sum(len(m['content']) for m in msgs[-n:]) // 4


def comp_tok(msgs):
    first = [m['content'] for m in msgs[:2]]
    last = [m['content'] for m in msgs[-5:]]
    sents = []
    for m in msgs[2:-5]:
        for s in re.split(r'[.!?]\s+', m['content']):
            if len(s) > 30:
                n = len(re.findall(r'[A-Z][a-z]{2,}|\d+\.?\d*', s))
                sents.append((n, s.strip()))
    sents.sort(reverse=True)
    key = [s for _, s in sents[:5]]
    return (sum(len(m['content']) for m in msgs[:2]) +
            sum(len(s) for s in key) +
            sum(len(m['content']) for m in msgs[-5:])) // 4


def extr_tok(msgs, q):
    sents = []
    for m in msgs:
        for s in re.split(r'[.!?]\s+', m['content']):
            if len(s) > 15:
                sents.append(s.strip())
    qw = set(q.lower().split())
    matching = [s for s in sents if any(w in s.lower() for w in qw)][:8]
    return sum(len(s) for s in matching) // 4


# ============ RUN ============
for ds_name, msgs, gts in [
    ('BACKEND (30msg, 918tok)', t1.CONVERSATION, t1.GROUND_TRUTH),
    ('FRONTEND (40msg, 1813tok)', t2.CONVERSATION, t2.GROUND_TRUTH),
]:
    orig_tok = sum(len(m['content']) for m in msgs) // 4
    full = ' '.join(m['content'] for m in msgs)

    # Get LLM anchor tokens
    g_llm = extract_graph(msgs)
    ac_tok_llm = g_llm.total_chars // 4

    # Get fallback anchor tokens
    old_key = os.environ.pop('DEEPSEEK_API_KEY', None)
    g_fb = extract_graph(msgs)
    if old_key:
        os.environ['DEEPSEEK_API_KEY'] = old_key
    ac_tok_fb = g_fb.total_chars // 4

    # Score all methods
    all_scores = {}

    # 1. Full conversation
    all_scores['FULL CONVERSATION'] = [score(full, gt) for gt in gts.values()]

    # 2. Sliding window
    sw_text = slide_text(msgs, 0.25)
    all_scores['SLIDING WINDOW'] = [score(sw_text, gt) for gt in gts.values()]

    # 3. Compaction sim
    comp_sim_text = comp_text(msgs)
    all_scores['COMPACTION SIM'] = [score(comp_sim_text, gt) for gt in gts.values()]

    # 4. Extractive keywords
    ek_scores = []
    ek_toks = []
    for q, gt in gts.items():
        t = extr_text(msgs, q)
        ek_scores.append(score(t, gt))
        ek_toks.append(extr_tok(msgs, q))
    all_scores['EXTRACTIVE KEYWORDS'] = ek_scores

    # 5. Anchor - Fallback
    af_scores = []
    for q, gt in gts.items():
        t = anchor_text(msgs, q, use_llm=False)
        af_scores.append(score(t, gt) if t else 0)
    all_scores['ANCHOR (fallback)'] = af_scores

    # 6. Anchor - LLM
    al_scores = []
    for q, gt in gts.items():
        t = anchor_text(msgs, q, use_llm=True)
        al_scores.append(score(t, gt) if t else 0)
    all_scores['ANCHOR (LLM)'] = al_scores

    # Print summary
    print(f"\n{'='*75}")
    print(f"  {ds_name}")
    print(f"{'='*75}")

    methods_meta = [
        ('FULL CONVERSATION', orig_tok, '$0'),
        ('SLIDING WINDOW', slide_tok(msgs, 0.25), '$0'),
        ('COMPACTION SIM', comp_tok(msgs), '$0'),
        ('EXTRACTIVE KEYWORDS', sum(ek_toks)//len(ek_toks), '$0'),
        ('ANCHOR (fallback)', ac_tok_fb, '$0'),
        ('ANCHOR (LLM)', ac_tok_llm, '$0.001'),
    ]

    print(f"  {'Method':<30s} {'Avg':>5s} {'Tok':>6s} {'Comp':>6s} {'Cost':>7s}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")

    for mname, mtok, mcost in methods_meta:
        scores = all_scores[mname]
        avg = sum(scores) / len(scores)
        comp = (1 - mtok / orig_tok) * 100
        print(f"  {mname:<30s} {avg:>4.1f}  {mtok:>5d}  {comp:>5.0f}%  {mcost:>7s}")

    # Per-query breakdown
    queries = list(gts.keys())
    print(f"\n  {'Query':<45s} {'Full':>4s} {'Slid':>4s} {'CmpS':>4s} {'ExtK':>4s} {'A-FB':>4s} {'A-LLM':>5s}")
    print(f"  {'-'*45} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*5}")
    for i, q in enumerate(queries):
        fs = all_scores['FULL CONVERSATION'][i]
        sw = all_scores['SLIDING WINDOW'][i]
        cs = all_scores['COMPACTION SIM'][i]
        ek = all_scores['EXTRACTIVE KEYWORDS'][i]
        af = all_scores['ANCHOR (fallback)'][i]
        al = all_scores['ANCHOR (LLM)'][i]
        print(f"  {q[:44]:<45s} {fs:>3.0f}  {sw:>3.0f}  {cs:>3.0f}  {ek:>3.0f}  {af:>4.0f}  {al:>5.0f}")

print()
