#!/usr/bin/env python3
"""
search_index.py (Developer Flavor, disk-based, with optimizations)

Usage:
  python search_index.py OUT_DIR

Requires outputs from build_index.py:
  - docinfo.json
  - lexicon.json + postings.bin
  - lexicon2.json + postings2.bin (optional but recommended for 2-gram)

Ranking:
  - Unigram cosine(tf-idf), idf = log10(N/df)
  - Bigram cosine(tf-idf) using 2-gram index
  - Positional phrase boost from unigram positions

Optimizations:
  - phrase_boost 仅对 unigram 得分最高的前 PHRASE_DOC_LIMIT 个文档做位置检查
  - 最终参与合成打分的候选文档数上限为 MAX_CANDIDATES

打印每个查询的响应时间 (ms)。
"""

import sys
import json
import math
import heapq
import time
from collections import Counter, defaultdict

# 复用 build_index.py 里的 tokenizer 和 decode_positions
from build_index import tokenize, decode_positions

# ---------- 超参数 / 权重 (可以按需微调) ----------
ALPHA_UNI = 0.85        # unigram cosine 权重
BETA_BI = 0.15          # bigram cosine 权重
PHRASE_BOOST = 0.15     # 每次短语命中的加分

TOPK_DEFAULT = 10       # 每次显示前多少个结果
MIN_RESULTS = 20        # 至少希望有这么多结果（用于多阶段候选）
PHRASE_DOC_LIMIT = 2000 # phrase_boost 最多处理的 doc 数（按 unigram score 排前 N）
MAX_CANDIDATES = 3000   # 最终参与合成打分的候选 doc 上限


# ---------- 读取元数据 ----------
def load_meta(out_dir):
    with open(f"{out_dir}/docinfo.json", "r", encoding="utf-8") as f:
        docinfo = {int(k): v for k, v in json.load(f).items()}

    with open(f"{out_dir}/lexicon.json", "r", encoding="utf-8") as f:
        lex1 = json.load(f)

    try:
        with open(f"{out_dir}/lexicon2.json", "r", encoding="utf-8") as f:
            lex2 = json.load(f)
    except Exception:
        lex2 = {}

    return docinfo, lex1, lex2


# ---------- 从磁盘读取 postings ----------
def read_postings(fp, lexicon, term):
    """
    返回 (df, postings_list) 或 None。
    postings_list 格式：
      - unigram: [[doc_id, tfw, posb64], ...]
      - bigram : [[doc_id, tf], ...]
    """
    entry = lexicon.get(term)
    if not entry:
        return None
    df, offset, nbytes = entry
    fp.seek(offset)
    rec = fp.read(nbytes).decode("utf-8")
    # 形如: term \t jsonpayload \n
    _, payload = rec.rstrip("\n").split("\t", 1)
    return df, json.loads(payload)


# ---------- 位置辅助函数 ----------
def count_adjacent(pos1, pos2):
    """统计 pos1 中某位置 +1 是否在 pos2 中（两列表均升序）。"""
    i = j = 0
    cnt = 0
    while i < len(pos1) and j < len(pos2):
        a = pos1[i] + 1
        b = pos2[j]
        if a == b:
            cnt += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return cnt


# ---------- unigram tf-idf + cosine ----------
def unigram_cosine(query_terms, docinfo, lex1, out_dir):
    """
    返回:
      uni_scores: doc -> cosine 分数
      pos_str_by_term: term -> {doc: posb64}  (供 phrase_boost 使用)
      term_docs: term -> set(doc_id)         (供 Boolean AND / 候选集使用)
    """
    N = len(docinfo)
    q_tf = Counter(query_terms)

    q_w = {}
    term_df = {}
    postings_cache = {}
    pos_str_by_term = defaultdict(dict)
    term_docs = defaultdict(set)

    with open(f"{out_dir}/postings.bin", "rb") as pf:
        # 先为 query 中的每个 term 取 postings + 计算 query 向量权重
        for t, tfq in q_tf.items():
            got = read_postings(pf, lex1, t)
            if not got:
                continue
            df, plist = got
            term_df[t] = df
            postings_cache[t] = plist

            # 记录 term -> 文档集合（用于 Boolean AND）
            docs_for_t = set()
            for doc, tfw, posb64 in plist:
                docs_for_t.add(doc)
                if posb64:
                    pos_str_by_term[t][doc] = posb64
            term_docs[t] = docs_for_t

            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            q_w[t] = (1.0 + math.log10(tfq)) * idf

        if not q_w:
            return {}, {}, term_docs

        # query 向量范数
        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        dot = defaultdict(float)

        # 累加 doc 向量和 query 向量的点积
        for t, w_tq in q_w.items():
            df = term_df[t]
            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue

            plist = postings_cache[t]
            for doc, tfw, _ in plist:
                if tfw <= 0:
                    continue
                w_td = (1.0 + math.log10(tfw)) * idf
                dot[doc] += w_td * w_tq

        uni_scores = {}
        for doc, num in dot.items():
            d_norm = docinfo.get(doc, {}).get("norm", 1.0) or 1.0
            uni_scores[doc] = num / (d_norm * q_norm)

        return uni_scores, pos_str_by_term, term_docs


# ---------- bigram tf-idf + cosine ----------
def bigram_cosine(query_terms, docinfo, lex2, out_dir):
    if not lex2:
        return {}

    # 构造 query 的 bigram 序列
    bigrams = []
    for i in range(len(query_terms) - 1):
        bigrams.append(query_terms[i] + " " + query_terms[i + 1])
    if not bigrams:
        return {}

    N = len(docinfo)
    q_tf = Counter(bigrams)

    q_w = {}
    cache = {}

    with open(f"{out_dir}/postings2.bin", "rb") as pf:
        for bg, tfq in q_tf.items():
            got = read_postings(pf, lex2, bg)
            if not got:
                continue
            df, plist = got
            cache[bg] = (df, plist)

            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            q_w[bg] = (1.0 + math.log10(tfq)) * idf

        if not q_w:
            return {}

        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        dot = defaultdict(float)

        for bg, w_tq in q_w.items():
            df, plist = cache[bg]
            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            for doc, tf in plist:
                if tf <= 0:
                    continue
                w_td = (1.0 + math.log10(tf)) * idf
                dot[doc] += w_td * w_tq

        bi_scores = {}
        for doc, num in dot.items():
            d_norm2 = docinfo.get(doc, {}).get("norm2", 1.0) or 1.0
            bi_scores[doc] = num / (d_norm2 * q_norm)

        return bi_scores


# ---------- phrase boost（带 doc 上限） ----------
def phrase_boost(query_terms, pos_str_by_term, allowed_docs=None):
    """
    对相邻 query term (t_i, t_{i+1}) 统计 body 中相邻出现次数，
    每次命中给该 doc 加 PHRASE_BOOST。

    allowed_docs: 若非 None，仅对其中 doc 做 boost（用于限制工作量）。
    """
    if len(query_terms) < 2:
        return {}

    boost = defaultdict(float)
    pairs = [(query_terms[i], query_terms[i + 1]) for i in range(len(query_terms) - 1)]

    for t1, t2 in pairs:
        docs1 = pos_str_by_term.get(t1, {})
        docs2 = pos_str_by_term.get(t2, {})
        if not docs1 or not docs2:
            continue

        common = set(docs1.keys()) & set(docs2.keys())
        if allowed_docs is not None:
            common &= allowed_docs
        if not common:
            continue

        for doc in common:
            p1 = decode_positions(docs1[doc])
            p2 = decode_positions(docs2[doc])
            hits = count_adjacent(p1, p2)
            if hits:
                boost[doc] += PHRASE_BOOST * hits

    return boost


# ---------- Boolean AND（基于 term_docs） ----------
def boolean_and_candidates(query_terms, term_docs):
    """
    对 query_terms 做 AND：每个 term 都出现的文档集合。
    term_docs: term -> set(doc_id)
    """
    posting_sets = []
    for t in query_terms:
        s = term_docs.get(t)
        if not s:
            return set()
        posting_sets.append(s)

    docs = posting_sets[0]
    for s in posting_sets[1:]:
        docs = docs & s
        if not docs:
            break
    return docs


# ---------- 交互查询 ----------
def interactive(out_dir, topk=TOPK_DEFAULT):
    docinfo, lex1, lex2 = load_meta(out_dir)

    print("Disk search: tf-idf + cosine (idf=log10(N/df)) + bigram + positional phrase boost")
    print(f"(phrase_boost top {PHRASE_DOC_LIMIT} docs, max candidates {MAX_CANDIDATES})")
    print("Empty line quits.")
    print("--------------------------------------------------------------")

    while True:
        try:
            q = input("Query> ").strip()
        except EOFError:
            break
        if not q:
            break

        t0 = time.perf_counter()

        q_terms = tokenize(q)
        if not q_terms:
            t1 = time.perf_counter()
            print(f"No valid terms.  (time={(t1 - t0) * 1000:.2f} ms)\n")
            continue

        # 1) unigram 主分数 + 记录 positions / term_docs
        uni_scores, pos_str_by_term, term_docs = unigram_cosine(q_terms, docinfo, lex1, out_dir)

        if not uni_scores:
            t1 = time.perf_counter()
            print(f"No results (no unigram hits). (time={(t1 - t0) * 1000:.2f} ms)\n")
            continue

        # 为 phrase_boost 构造允许的 doc 集：按 unigram 分数取前 PHRASE_DOC_LIMIT 个
        if len(uni_scores) > PHRASE_DOC_LIMIT:
            top_for_phrase = heapq.nlargest(PHRASE_DOC_LIMIT, uni_scores.items(), key=lambda x: x[1])
            phrase_allowed = {d for d, _ in top_for_phrase}
        else:
            phrase_allowed = set(uni_scores.keys())

        # 2) bigram 辅助分数
        bi_scores = bigram_cosine(q_terms, docinfo, lex2, out_dir)

        # 3) phrase boost（仅对 phrase_allowed 的 doc）
        boosts = phrase_boost(q_terms, pos_str_by_term, allowed_docs=phrase_allowed)

        # 4) 多阶段候选选择
        all_docs_base = set(uni_scores.keys()) | set(bi_scores.keys()) | set(boosts.keys())

        # Stage 1: 有 phrase 的 doc 优先
        phrase_docs = {d for d, b in boosts.items() if b > 0.0}
        if len(phrase_docs) >= MIN_RESULTS:
            candidate_docs = phrase_docs
        else:
            # Stage 2: Boolean AND
            and_docs = boolean_and_candidates(q_terms, term_docs)
            candidate_docs = phrase_docs | and_docs

            if len(candidate_docs) < MIN_RESULTS:
                # Stage 3: fallback 到所有向量空间候选
                candidate_docs = all_docs_base

        # 限制候选 doc 数量，避免在尾部 doc 上浪费太多计算
        if len(candidate_docs) > MAX_CANDIDATES:
            # 按 unigram 分数截断：没在 uni_scores 里的 doc 得分视作 0
            top_candidates = heapq.nlargest(
                MAX_CANDIDATES,
                candidate_docs,
                key=lambda d: uni_scores.get(d, 0.0)
            )
            candidate_docs = set(top_candidates)

        # 5) 只对 candidate_docs 合成最终分数
        combined = {}
        for d in candidate_docs:
            combined[d] = (
                ALPHA_UNI * uni_scores.get(d, 0.0)
                + BETA_BI  * bi_scores.get(d, 0.0)
                + boosts.get(d, 0.0)
            )

        if not combined:
            t1 = time.perf_counter()
            print(f"No results. (time={(t1 - t0) * 1000:.2f} ms)\n")
            continue

        top = heapq.nlargest(topk, combined.items(), key=lambda x: x[1])

        t1 = time.perf_counter()

        for i, (doc, score) in enumerate(top, 1):
            url = docinfo.get(doc, {}).get("url", f"doc:{doc}")
            print(f"{i}. {score:.6f}  {url}")

        print(f"(response time: {(t1 - t0) * 1000:.2f} ms, candidates={len(candidate_docs)})\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python search_index.py OUT_DIR")
        sys.exit(1)
    interactive(sys.argv[1])


if __name__ == "__main__":
    main()
