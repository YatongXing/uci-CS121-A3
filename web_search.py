#!/usr/bin/env python3
"""
web_search.py

Simple web interface for your search engine (Milestone 3 extra credit).

Usage:
    python3 web_search.py out_index

where out_index is the directory produced by build_index.py
containing:
  - docinfo.json
  - lexicon.json, postings.bin
  - lexicon2.json, postings2.bin

open http://127.0.0.1:5000/ in local browser
"""

import sys
import heapq

from flask import Flask, request, render_template_string

# Reuse tokenizer and all search helpers
from build_index import tokenize
import search_index as si


# ---------- command-line args & global data ----------

if len(sys.argv) != 2:
    print("Usage: python3 web_search.py OUT_DIR")
    sys.exit(1)

OUT_DIR = sys.argv[1]

print(f"Loading metadata from {OUT_DIR} ...")
docinfo, lex1, lex2 = si.load_meta(OUT_DIR)
print(f"Loaded {len(docinfo)} documents.")


# ---------- Flask app ----------

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CS121 Search Engine</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
           margin: 2rem; background: #f6f6f6; }
    h1 { margin-bottom: 0.5rem; }
    form { margin-bottom: 1.5rem; }
    input[type="text"] {
        width: 60%;
        padding: 0.5rem 0.75rem;
        border-radius: 20px;
        border: 1px solid #ccc;
        font-size: 1rem;
    }
    button {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: none;
        background: #0077ff;
        color: white;
        font-size: 1rem;
        cursor: pointer;
        margin-left: 0.5rem;
    }
    button:hover { background: #005fd1; }
    .result {
        background: white;
        margin-bottom: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .result-score {
        color: #666;
        font-size: 0.9rem;
    }
    .no-results {
        color: #c00;
        font-style: italic;
    }
    .footer {
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #777;
    }
  </style>
</head>
<body>
  <h1>CS121 Search Engine</h1>
  <form method="get" action="/">
    <input type="text" name="q" placeholder="Type your query..."
           value="{{ query|e }}" autofocus>
    <button type="submit">Search</button>
  </form>

  {% if query %}
    {% if results %}
      <p>Showing top {{ results|length }} results for <b>{{ query|e }}</b>:</p>
      {% for r in results %}
        <div class="result">
          <div><a href="{{ r.url }}" target="_blank">{{ r.url }}</a></div>
          <div class="result-score">
            doc_id = {{ r.doc_id }}, score = {{ "%.6f"|format(r.score) }}
          </div>
        </div>
      {% endfor %}
    {% else %}
      <p class="no-results">No results for "{{ query|e }}".</p>
    {% endif %}
  {% endif %}

  <div class="footer">
    Backend: disk-based tf-idf cosine + bigram + phrase boost (same as CLI).
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def search_page():
    query = (request.args.get("q") or "").strip()
    results = []

    if query:
        q_terms = tokenize(query)

        if q_terms:
            # ---- 1) unigram scores + position info + term_docs ----
            uni_scores, pos_str_by_term, term_docs = si.unigram_cosine(q_terms, docinfo, lex1, OUT_DIR)

            if uni_scores:
                # For phrase boost we follow your interactive() logic:

                # phrase_boost is only computed on top PHRASE_DOC_LIMIT unigram docs
                if len(uni_scores) > si.PHRASE_DOC_LIMIT:
                    top_for_phrase = heapq.nlargest(
                        si.PHRASE_DOC_LIMIT, uni_scores.items(), key=lambda x: x[1]
                    )
                    phrase_allowed = {d for d, _ in top_for_phrase}
                else:
                    phrase_allowed = set(uni_scores.keys())

                # ---- 2) bigram scores ----
                bi_scores = si.bigram_cosine(q_terms, docinfo, lex2, OUT_DIR)

                # ---- 3) phrase boost (restricted docs) ----
                boosts = si.phrase_boost(
                    q_terms, pos_str_by_term, allowed_docs=phrase_allowed
                )

                # ---- 4) multi-stage candidate selection ----
                all_docs_base = set(uni_scores.keys()) | set(bi_scores.keys()) | set(boosts.keys())
                phrase_docs = {d for d, b in boosts.items() if b > 0.0}

                if len(phrase_docs) >= si.MIN_RESULTS:
                    candidate_docs = phrase_docs
                else:
                    and_docs = si.boolean_and_candidates(q_terms, term_docs)
                    candidate_docs = phrase_docs | and_docs

                    if len(candidate_docs) < si.MIN_RESULTS:
                        candidate_docs = all_docs_base

                if len(candidate_docs) > si.MAX_CANDIDATES:
                    top_candidates = heapq.nlargest(
                        si.MAX_CANDIDATES,
                        candidate_docs,
                        key=lambda d: uni_scores.get(d, 0.0),
                    )
                    candidate_docs = set(top_candidates)

                # ---- 5) combine scores only for candidates ----
                combined = {}
                for d in candidate_docs:
                    combined[d] = (
                        si.ALPHA_UNI * uni_scores.get(d, 0.0)
                        + si.BETA_BI  * bi_scores.get(d, 0.0)
                        + boosts.get(d, 0.0)
                    )

                top = heapq.nlargest(20, combined.items(), key=lambda x: x[1])

                for doc_id, score in top:
                    url = docinfo.get(doc_id, {}).get("url", f"doc:{doc_id}")
                    results.append({
                        "doc_id": doc_id,
                        "score": score,
                        "url": url,
                    })

    return render_template_string(PAGE_TEMPLATE, query=query, results=results)


if __name__ == "__main__":
    # debug=True is convenient while you're developing
    app.run(debug=True)
