#!/usr/bin/env python3
"""
web_search.py

Simple web interface for the search engine (Milestone 3 extra credit).

Usage:
    python3 web_search.py OUT_DIR

OUT_DIR is the directory produced by build_index.py, containing:
  - docinfo.json
  - lexicon.json, postings.bin
  - lexicon2.json, postings2.bin


open browser in http://127.0.0.1:5000/
"""

import sys

from flask import Flask, request, render_template_string

# Reuse tokenizer + search pipeline
from build_index import tokenize
import search_index as si   # use si.load_meta, si.search_one_pass, etc.

# ---------- CLI args + global data ----------

if len(sys.argv) != 2:
    print("Usage: python3 web_search.py OUT_DIR")
    sys.exit(1)

OUT_DIR = sys.argv[1]

print(f"Loading metadata from {OUT_DIR} ...")
DOCINFO, LEX1, LEX2 = si.load_meta(OUT_DIR)
print(f"Loaded {len(DOCINFO)} documents.")

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
      <p>Showing {{ results|length }} results for <b>{{ query|e }}</b>:</p>
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
    Backend: disk-based tf-idf cosine + bigram + phrase boost (same logic as CLI).
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
            # Use your single-pass search pipeline
            top, cand_count = si.search_one_pass(
                q_terms, DOCINFO, LEX1, LEX2, OUT_DIR
            )

            if top:  # list of (doc_id, score)
                for doc_id, score in top:
                    url = DOCINFO.get(doc_id, {}).get("url", f"doc:{doc_id}")
                    results.append({
                        "doc_id": doc_id,
                        "score": score,
                        "url": url,
                    })

    return render_template_string(PAGE_TEMPLATE, query=query, results=results)


if __name__ == "__main__":
    # debug=True is convenient while developing;
    app.run(debug=False)
