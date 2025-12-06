#!/usr/bin/env python3
"""

Description:
  This script performs ranked retrieval on the inverted index built by build_index.py.
  It implements a "Champion List" style retrieval with a tiered scoring strategy:
  1. Unigram Cosine Similarity (Vector Space Model).
  2. Bigram Cosine Similarity (for better local context).
  3. Phrase Proximity Boosting (checking exact adjacent positions).

  Optimizations:
  - Memory: Keeps the Lexicon in RAM, but reads Postings from disk on-demand.
  - Latency: Disables Python Garbage Collection (GC) during the search phase
    to prevent random pauses during query processing.
  - Pruning: Limits the number of candidate documents processed to ensure speed.

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

# Import dependencies from build_index to ensure consistent tokenization logic
from build_index import tokenize, decode_positions

# ---------- Hyperparameters ----------
# Weights for mixing Unigram and Bigram scores (should sum to ~1.0)
ALPHA_UNI = 0.85
BETA_BI = 0.15

# Bonus score added for every occurrence of an exact phrase match
PHRASE_BOOST = 0.15

# Search Limits / Pruning
TOPK_DEFAULT = 10  # Number of final results to display
MIN_RESULTS = 50  # Minimum candidates to consider before relaxing constraints
PHRASE_DOC_LIMIT = 5000  # Max docs to check for expensive phrase position matching
MAX_CANDIDATES = 5000  # Hard cap on number of documents fully scored


# Metadata Loading
def load_meta(out_dir):
    """
    Loads index metadata into memory.
    - docinfo: URL, norms, duplicate info.
    - lexicon: Map of term -> [doc_freq, byte_offset, byte_length].
    """
    with open(f"{out_dir}/docinfo.json", "r", encoding="utf-8") as f:
        # Convert string keys back to integers for doc_ids
        docinfo = {int(k): v for k, v in json.load(f).items()}

    with open(f"{out_dir}/lexicon.json", "r", encoding="utf-8") as f:
        lex1 = json.load(f)

    # Load bigram lexicon if it exists
    try:
        with open(f"{out_dir}/lexicon2.json", "r", encoding="utf-8") as f:
            lex2 = json.load(f)
    except Exception:
        lex2 = {}

    return docinfo, lex1, lex2


# Disk Reading
def read_postings(fp, lexicon, term):
    """
    Random Access Reader:
    Seeks to the specific byte offset in the large binary file to read
    only the posting list for the requested term.
    Returns: (Document Frequency, List of Postings)
    """
    entry = lexicon.get(term)
    if not entry:
        return None
    df, offset, nbytes = entry

    fp.seek(offset)
    rec = fp.read(nbytes).decode("utf-8")

    # Format on disk: "term \t JSON_LIST"
    _, payload = rec.rstrip("\n").split("\t", 1)
    return df, json.loads(payload)


# Utils
def count_adjacent(pos1, pos2):
    """
    Merges two sorted position lists to find exact adjacency.
    Returns count of times term2 follows term1 immediately (p2 == p1 + 1).
    Used for Phrase Boosting.
    """
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
    """
    Returns the intersection of document sets for all query terms (AND query).
    Used as a fallback filter if phrase search yields too few results.
    """
    posting_sets = []
    for t in query_terms:
        s = term_docs.get(t)
        if not s:
            return set()  # One term missing implies intersection is empty
        posting_sets.append(s)

    # Start with first set and intersect with others
    docs = posting_sets[0]
    for s in posting_sets[1:]:
        docs = docs & s
        if not docs:
            break
    return docs


# Scoring Logic
def unigram_cosine(query_terms, docinfo, lex1, out_dir):
    """
    Calculates TF-IDF Cosine Similarity for single terms.

    Process:
    1. Calculate Query Vector Weight (q_w).
    2. Retrieve postings for all query terms.
    3. Calculate Dot Product: sum(w_query * w_doc) for each doc.
    4. Normalize: dot_product / (doc_norm * query_norm).

    Returns:
       - uni_scores: dict {doc_id: score}
       - pos_str_by_term: dict {term: {doc: encoded_positions}} (for phrase check)
       - term_docs: dict {term: set(doc_ids)} (for boolean logic)
    """
    N = len(docinfo)
    q_tf = Counter(query_terms)
    q_w = {}
    term_df = {}
    postings_cache = {}
    pos_str_by_term = defaultdict(dict)
    term_docs = defaultdict(set)

    with open(f"{out_dir}/postings.bin", "rb") as pf:
        # Step 1: Fetch data and build Query Vector
        for t, tfq in q_tf.items():
            got = read_postings(pf, lex1, t)
            if not got:
                continue
            df, plist = got
            term_df[t] = df
            postings_cache[t] = plist

            # Cache doc sets and positions for later steps
            docs_for_t = set()
            for doc, tfw, posb64 in plist:
                docs_for_t.add(doc)
                if posb64:
                    pos_str_by_term[t][doc] = posb64
            term_docs[t] = docs_for_t

            # W_tq = (1 + log(tf)) * idf
            idf = math.log10(N / df) if df > 0 else 0.0
            if idf == 0.0:
                continue
            q_w[t] = (1.0 + math.log10(tfq)) * idf

        if not q_w:
            return {}, {}, term_docs

        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        dot = defaultdict(float)

        # Step 2: Accumulate Dot Product
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

        # Step 3: Normalize Scores
        uni_scores = {}
        for doc, num in dot.items():
            d_norm = docinfo.get(doc, {}).get("norm", 1.0) or 1.0
            uni_scores[doc] = num / (d_norm * q_norm)

        return uni_scores, pos_str_by_term, term_docs


def bigram_cosine(query_terms, docinfo, lex2, out_dir):
    """
    Calculates TF-IDF Cosine Similarity for Bigrams (2-word sequences).
    This helps capture local context (e.g., "new york" vs "new" and "york").
    """
    if not lex2:
        return {}

    # Generate bigrams from query: ["hello", "world"] -> ["hello world"]
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
    """
    Checks for exact phrase adjacency in documents.
    Adds a static score boost (PHRASE_BOOST) for every phrase occurrence found.
    Optimization: Only checks documents in 'allowed_docs' to save CPU.
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

        # Find docs containing both terms
        common = set(docs1.keys()) & set(docs2.keys())
        if allowed_docs is not None:
            common &= allowed_docs
        if not common:
            continue

        for doc in common:
            # Decode positions from Base64/VarInt
            p1 = decode_positions(docs1[doc])
            p2 = decode_positions(docs2[doc])

            # Check adjacency
            hits = count_adjacent(p1, p2)
            if hits:
                boost[doc] += PHRASE_BOOST * hits
    return boost


def search_one_pass(q_terms, docinfo, lex1, lex2, out_dir):
    """
    Orchestrates the search process.
    Returns top K results.
    """
    # 1. Calculate Unigram Scores (Base relevance)
    uni_scores, pos_str_by_term, term_docs = unigram_cosine(q_terms, docinfo, lex1, out_dir)
    if not uni_scores:
        return None, 0

    # Pruning: Only check phrases for the top N unigram documents to save time
    if len(uni_scores) > PHRASE_DOC_LIMIT:
        top_for_phrase = heapq.nlargest(PHRASE_DOC_LIMIT, uni_scores.items(), key=lambda x: x[1])
        phrase_allowed = {d for d, _ in top_for_phrase}
    else:
        phrase_allowed = set(uni_scores.keys())

    # 2. Calculate Bigram Scores
    bi_scores = bigram_cosine(q_terms, docinfo, lex2, out_dir)

    # 3. Calculate Phrase Boosts (Proximity)
    boosts = phrase_boost(q_terms, pos_str_by_term, allowed_docs=phrase_allowed)

    # 4. Candidate Selection (Tiered Strategy)
    # Priority: Docs with phrase match > Docs with all terms (AND) > Any Doc (OR)
    all_docs_base = set(uni_scores.keys()) | set(bi_scores.keys()) | set(boosts.keys())
    phrase_docs = {d for d, b in boosts.items() if b > 0.0}

    if len(phrase_docs) >= MIN_RESULTS:
        candidate_docs = phrase_docs
    else:
        # If not enough phrase matches, try AND logic
        and_docs = boolean_and_candidates(q_terms, term_docs)
        candidate_docs = phrase_docs | and_docs
        # If still not enough, fallback to OR (everything)
        if len(candidate_docs) < MIN_RESULTS:
            candidate_docs = all_docs_base

    # Hard cap on candidates to ensure sub-second response
    if len(candidate_docs) > MAX_CANDIDATES:
        top_candidates = heapq.nlargest(
            MAX_CANDIDATES, candidate_docs, key=lambda d: uni_scores.get(d, 0.0)
        )
        candidate_docs = set(top_candidates)

    # 5. Final Combined Score Calculation
    combined = {}
    for d in candidate_docs:
        combined[d] = (
                ALPHA_UNI * uni_scores.get(d, 0.0)
                + BETA_BI * bi_scores.get(d, 0.0)
                + boosts.get(d, 0.0)
        )

    if not combined:
        return None, 0

    # Retrieve Top K
    top = heapq.nlargest(TOPK_DEFAULT, combined.items(), key=lambda x: x[1])
    return top, len(candidate_docs)


# Interactive Loop
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

        # Performance Optimization
        # Python's Garbage Collector can trigger randomly, causing latency spikes.
        # We manually collect before search, then disable it during the critical path.
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            top, count = search_one_pass(q_terms, docinfo, lex1, lex2, out_dir)
            t1 = time.perf_counter()
        finally:
            gc.enable()  # Re-enable GC

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