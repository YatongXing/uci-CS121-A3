#!/usr/bin/env python3
"""
search_index.py (Final Version - Single Run with GC Control)

Usage:
  python search_index.py OUT_DIR
"""

import sys
import json
import math
import heapq
import time
import gc
from collections import Counter, defaultdict

# Import dependencies from build_index
from build_index import tokenize, decode_positions

# ---------- Hyperparameters ----------
ALPHA_UNI = 0.85
BETA_BI = 0.15
PHRASE_BOOST = 0.15

TOPK_DEFAULT = 10
MIN_RESULTS = 50
PHRASE_DOC_LIMIT = 5000
MAX_CANDIDATES = 5000

# ---------- Metadata Loading ----------
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

# ---------- Disk Reading ----------
def read_postings(fp, lexicon, term):
    entry = lexicon.get(term)
    if not entry:
        return None
    df, offset, nbytes = entry
    fp.seek(offset)
    rec = fp.read(nbytes).decode("utf-8")
    _, payload = rec.rstrip("\n").split("\t", 1)
    return df, json.loads(payload)

# ---------- Utils ----------
def count_adjacent(pos1, pos2):
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

def boolean_and_candidates(query_terms, term_docs):
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

# ---------- Scoring Logic ----------
def unigram_cosine(query_terms, docinfo, lex1, out_dir):
    N = len(docinfo)
    q_tf = Counter(query_terms)
    q_w = {}
    term_df = {}
    postings_cache = {}
    pos_str_by_term = defaultdict(dict)
    term_docs = defaultdict(set)

    with open(f"{out_dir}/postings.bin", "rb") as pf:
        for t, tfq in q_tf.items():
            got = read_postings(pf, lex1, t)
            if not got:
                continue
            df, plist = got
            term_df[t] = df
            postings_cache[t] = plist

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

        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        dot = defaultdict(float)

        for t, w_tq in q_w.items():
            df = term_df[t]
            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            for doc, tfw, _ in postings_cache[t]:
                if tfw <= 0:
                    continue
                w_td = (1.0 + math.log10(tfw)) * idf
                dot[doc] += w_td * w_tq

        uni_scores = {}
        for doc, num in dot.items():
            d_norm = docinfo.get(doc, {}).get("norm", 1.0) or 1.0
            uni_scores[doc] = num / (d_norm * q_norm)

        return uni_scores, pos_str_by_term, term_docs

def bigram_cosine(query_terms, docinfo, lex2, out_dir):
    if not lex2:
        return {}
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

def phrase_boost(query_terms, pos_str_by_term, allowed_docs=None):
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

def search_one_pass(q_terms, docinfo, lex1, lex2, out_dir):
    """Executes one full search pass."""
    # 1. Unigram
    uni_scores, pos_str_by_term, term_docs = unigram_cosine(q_terms, docinfo, lex1, out_dir)
    if not uni_scores:
        return None, 0

    if len(uni_scores) > PHRASE_DOC_LIMIT:
        top_for_phrase = heapq.nlargest(PHRASE_DOC_LIMIT, uni_scores.items(), key=lambda x: x[1])
        phrase_allowed = {d for d, _ in top_for_phrase}
    else:
        phrase_allowed = set(uni_scores.keys())

    # 2. Bigram
    bi_scores = bigram_cosine(q_terms, docinfo, lex2, out_dir)

    # 3. Phrase Boost
    boosts = phrase_boost(q_terms, pos_str_by_term, allowed_docs=phrase_allowed)

    # 4. Candidates
    all_docs_base = set(uni_scores.keys()) | set(bi_scores.keys()) | set(boosts.keys())
    phrase_docs = {d for d, b in boosts.items() if b > 0.0}
    
    if len(phrase_docs) >= MIN_RESULTS:
        candidate_docs = phrase_docs
    else:
        and_docs = boolean_and_candidates(q_terms, term_docs)
        candidate_docs = phrase_docs | and_docs
        if len(candidate_docs) < MIN_RESULTS:
            candidate_docs = all_docs_base

    if len(candidate_docs) > MAX_CANDIDATES:
        top_candidates = heapq.nlargest(
            MAX_CANDIDATES, candidate_docs, key=lambda d: uni_scores.get(d, 0.0)
        )
        candidate_docs = set(top_candidates)

    # 5. Combined Score
    combined = {}
    for d in candidate_docs:
        combined[d] = (
            ALPHA_UNI * uni_scores.get(d, 0.0)
            + BETA_BI  * bi_scores.get(d, 0.0)
            + boosts.get(d, 0.0)
        )

    if not combined:
        return None, 0

    top = heapq.nlargest(TOPK_DEFAULT, combined.items(), key=lambda x: x[1])
    return top, len(candidate_docs)

# ---------- Interactive Loop ----------
def interactive(out_dir):
    docinfo, lex1, lex2 = load_meta(out_dir)

    print("Disk search: tf-idf + cosine + bigram + phrase boost")
    print("Optimization: Single run + GC disabled during search.")
    print("Empty line quits.")
    print("--------------------------------------------------------------")

    while True:
        try:
            q = input("Query> ").strip()
        except EOFError:
            break
        if not q:
            break

        q_terms = tokenize(q)
        if not q_terms:
            print(f"No valid terms.\n")
            continue

        # --- Single run, but still with GC control ---
        gc.collect() # Optional: clean up before starting
        gc.disable() # Disable GC for pure timing
        try:
            t0 = time.perf_counter()
            top, count = search_one_pass(q_terms, docinfo, lex1, lex2, out_dir)
            t1 = time.perf_counter()
        finally:
            gc.enable() # Re-enable GC

        elapsed = (t1 - t0) * 1000
        
        if not top:
            print(f"No results. (time={elapsed:.2f} ms)\n")
            continue

        for i, (doc, score) in enumerate(top, 1):
            url = docinfo.get(doc, {}).get("url", f"doc:{doc}")
            print(f"{i}. {score:.6f}  {url}")

        print(f"(response time: {elapsed:.2f} ms, candidates={count})\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python search_index.py OUT_DIR")
        sys.exit(1)
    interactive(sys.argv[1])

if __name__ == "__main__":
    main()