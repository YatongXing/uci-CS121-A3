#!/usr/bin/env python3
"""
search_index.py (Developer Flavor, disk-based) + timer

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

Prints response time (ms) for each query.
"""

import sys
import json
import math
import heapq
import time
from collections import Counter, defaultdict

# IMPORTANT: reuse same tokenizer and decode_positions from your build_index.py
# Make sure build_index.py contains tokenize() and decode_positions()
from build_index import tokenize, decode_positions

# ---------- score weights (tune if needed) ----------
ALPHA_UNI = 0.85      # unigram cosine weight
BETA_BI = 0.15        # bigram cosine weight
PHRASE_BOOST = 0.15   # additive boost per phrase occurrence (adjacent positions)
TOPK_DEFAULT = 10


# ---------- load docinfo + lexicons ----------
def load_meta(out_dir):
    with open(f"{out_dir}/docinfo.json", "r", encoding="utf-8") as f:
        docinfo = {int(k): v for k, v in json.load(f).items()}

    with open(f"{out_dir}/lexicon.json", "r", encoding="utf-8") as f:
        lex1 = json.load(f)

    # 2-gram index optional
    try:
        with open(f"{out_dir}/lexicon2.json", "r", encoding="utf-8") as f:
            lex2 = json.load(f)
    except Exception:
        lex2 = {}

    return docinfo, lex1, lex2


# ---------- disk postings read (seek+nbytes) ----------
def read_postings(fp, lexicon, term):
    """
    Returns (df, postings_list) or None.
    postings_list format (unigram): [[doc_id, tfw, posb64], ...]
    postings_list format (bigram):  [[doc_id, tf], ...]
    """
    entry = lexicon.get(term)
    if not entry:
        return None
    df, offset, nbytes = entry
    fp.seek(offset)
    rec = fp.read(nbytes).decode("utf-8")
    # record: term \t jsonpayload \n
    _, payload = rec.rstrip("\n").split("\t", 1)
    return df, json.loads(payload)


# ---------- positional phrase helpers ----------
def count_adjacent(pos1, pos2):
    """Count occurrences where p in pos1 and p+1 in pos2 (both sorted)."""
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


# ---------- unigram cosine(tf-idf) ----------
def unigram_cosine(query_terms, docinfo, lex1, out_dir):
    """
    Returns:
      uni_scores: doc -> cosine score
      pos_str_by_term: term -> {doc: posb64}   (for phrase boost)
    """
    N = len(docinfo)
    q_tf = Counter(query_terms)

    q_w = {}
    term_df = {}
    postings_cache = {}

    pos_str_by_term = defaultdict(dict)

    with open(f"{out_dir}/postings.bin", "rb") as pf:
        # compute query weights and cache postings
        for t, tfq in q_tf.items():
            got = read_postings(pf, lex1, t)
            if not got:
                continue
            df, plist = got
            term_df[t] = df
            postings_cache[t] = plist

            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            q_w[t] = (1.0 + math.log10(tfq)) * idf

        if not q_w:
            return {}, {}

        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        dot = defaultdict(float)

        for t, w_tq in q_w.items():
            df = term_df[t]
            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue

            for doc, tfw, posb64 in postings_cache[t]:
                if tfw <= 0:
                    continue
                w_td = (1.0 + math.log10(tfw)) * idf
                dot[doc] += w_td * w_tq
                if posb64:
                    pos_str_by_term[t][doc] = posb64

        uni_scores = {}
        for doc, num in dot.items():
            d_norm = docinfo.get(doc, {}).get("norm", 1.0) or 1.0
            uni_scores[doc] = num / (d_norm * q_norm)

        return uni_scores, pos_str_by_term


# ---------- bigram cosine(tf-idf) using 2-gram index ----------
def bigram_cosine(query_terms, docinfo, lex2, out_dir):
    if not lex2:
        return {}

    # build query bigrams in order
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


# ---------- phrase boost from positions ----------
def phrase_boost(query_terms, pos_str_by_term):
    """
    For each adjacent query term pair (t_i, t_{i+1}), count times they occur adjacently
    in the body positions. Add PHRASE_BOOST * hits to doc score (additive).
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
        for doc in common:
            p1 = decode_positions(docs1[doc])
            p2 = decode_positions(docs2[doc])
            hits = count_adjacent(p1, p2)
            if hits:
                boost[doc] += PHRASE_BOOST * hits

    return boost


# ---------- interactive loop + timing ----------
def interactive(out_dir, topk=TOPK_DEFAULT):
    docinfo, lex1, lex2 = load_meta(out_dir)

    print("Disk search: tf-idf + cosine (idf=log10(N/df)) + bigram + positional phrase boost")
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

        uni_scores, pos_str = unigram_cosine(q_terms, docinfo, lex1, out_dir)
        bi_scores = bigram_cosine(q_terms, docinfo, lex2, out_dir)
        boosts = phrase_boost(q_terms, pos_str)

        all_docs = set(uni_scores.keys()) | set(bi_scores.keys()) | set(boosts.keys())
        if not all_docs:
            t1 = time.perf_counter()
            print(f"No results.  (time={(t1 - t0) * 1000:.2f} ms)\n")
            continue

        combined = {}
        for d in all_docs:
            combined[d] = (
                ALPHA_UNI * uni_scores.get(d, 0.0)
                + BETA_BI * bi_scores.get(d, 0.0)
                + boosts.get(d, 0.0)
            )

        top = heapq.nlargest(topk, combined.items(), key=lambda x: x[1])

        t1 = time.perf_counter()

        for i, (doc, score) in enumerate(top, 1):
            url = docinfo.get(doc, {}).get("url", f"doc:{doc}")
            print(f"{i}. {score:.6f}  {url}")

        print(f"(response time: {(t1 - t0) * 1000:.2f} ms)\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python search_index.py OUT_DIR")
        sys.exit(1)
    interactive(sys.argv[1])


if __name__ == "__main__":
    main()
