#!/usr/bin/env python3
"""
search_index.py

Milestone 2: simple Boolean (AND-only) retrieval over the inverted index
built in Milestone 1.

Usage:
    python3 search_index.py index.json doc_ids.json

Then type queries at the prompt, e.g.
    cristina lopes
    machine learning
    ACM
    master of software engineering

Empty line quits.
"""

import sys
import json
import math
from pathlib import Path
from build_index import tokenize


# ---------- Loading index & doc-id map ----------

def load_index(index_path, doc_ids_path):
    """Load inverted index and doc_id -> file_path map from JSON."""
    with open(index_path, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    # JSON forces dict keys to strings; convert doc_ids back to int
    index = {}
    for term, postings in raw_index.items():
        index[term] = {int(doc_id): tf for doc_id, tf in postings.items()}

    with open(doc_ids_path, "r", encoding="utf-8") as f:
        raw_doc_map = json.load(f)
    doc_id_map = {int(doc_id): path for doc_id, path in raw_doc_map.items()}

    return index, doc_id_map


# ---------- Boolean AND retrieval + ranking ----------

def boolean_and_docs(index, query_terms):
    """
    Return the set of doc_ids that contain *all* query_terms.
    query_terms are already tokenized/stemmed.
    """
    posting_sets = []

    for term in query_terms:
        postings = index.get(term)
        if not postings:
            # no docs with this term -> AND query has no results
            return set()
        posting_sets.append(set(postings.keys()))

    # intersection of all posting sets
    docs = posting_sets[0]
    for s in posting_sets[1:]:
        docs = docs & s
        if not docs:
            break
    return docs


def rank_docs_tf_idf(index, query_terms, candidate_docs, N_docs):
    """
    Compute a simple tf-idf score for each candidate doc.
    score(d) = sum_{t in query} tf(t,d) * log(N / df(t))
    Returns a list of (doc_id, score) sorted by score desc.
    """
    if not candidate_docs:
        return []

    # pre-compute idf for terms that are actually in the index
    idf = {}
    for t in query_terms:
        postings = index.get(t)
        if postings:
            df = len(postings)
            # avoid division by zero; standard log-based idf
            idf[t] = math.log((N_docs + 1) / (df + 1)) + 1.0
        else:
            idf[t] = 0.0

    scores = {doc_id: 0.0 for doc_id in candidate_docs}

    for t in query_terms:
        postings = index.get(t, {})
        w = idf[t]
        if w == 0.0:
            continue
        for doc_id in candidate_docs:
            tf = postings.get(doc_id, 0)
            if tf:
                scores[doc_id] += tf * w

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


# ---------- Helper: get URL from a doc's JSON file ----------

def get_url_from_doc_file(doc_file_path):
    """
    Given the JSON file path stored in doc_id_map, open it and
    return the 'url' field if present, otherwise the file path.
    """
    try:
        with open(doc_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("url", doc_file_path)
    except Exception:
        # fall back to showing the path if something goes wrong
        return doc_file_path


# ---------- Interactive search loop ----------

def interactive_search(index, doc_id_map):
    N_docs = len(doc_id_map)

    print("Simple AND-only search. Empty line to quit.")
    print("-------------------------------------------")

    while True:
        try:
            query = input("Query> ").strip()
        except EOFError:
            break

        if not query:
            break

        # tokenize + stem exactly like MS1
        q_tokens = tokenize(query)
        if not q_tokens:
            print("No valid terms in query.")
            continue

        candidate_docs = boolean_and_docs(index, q_tokens)
        if not candidate_docs:
            print("No results.")
            continue

        ranked = rank_docs_tf_idf(index, q_tokens, candidate_docs, N_docs)

        print(f"Found {len(candidate_docs)} documents. Top 5:")
        for rank, (doc_id, score) in enumerate(ranked[:5], start=1):
            doc_path = doc_id_map[doc_id]
            url = get_url_from_doc_file(doc_path)
            print(f"{rank}. doc_id={doc_id}, score={score:.4f}")
            print(f"   URL: {url}")
        print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python search_index.py index.json doc_ids.json")
        sys.exit(1)

    index_path = Path(sys.argv[1])
    doc_ids_path = Path(sys.argv[2])

    index, doc_id_map = load_index(index_path, doc_ids_path)
    interactive_search(index, doc_id_map)


if __name__ == "__main__":
    main()
