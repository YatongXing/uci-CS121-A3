#!/usr/bin/env python3
"""
build_index.py

Milestone 1: build a simple in-memory inverted index over a corpus of HTML pages.

Usage:
    python3 build_index.py /path/to/DEV index.json doc_ids.json

- /path/to/DEV: root folder that contains many subfolders of HTML pages
- index.json:  output file where the inverted index will be stored
- doc_ids.json: output file with mapping from doc_id -> file_path
"""

import os
import re
import sys
import json
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
#   >>> import nltk; nltk.download('punkt')

# ---------- Tokenization / Normalization helpers ----------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

stemmer = PorterStemmer()


def extract_visible_text(html):
    """
    Parse (possibly broken) HTML and return visible text as a single string.
    We ignore scripts/styles etc.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements – their text is not useful for search.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    """
    Turn raw text into a list of normalized tokens:
    - alphanumeric sequences only
    - lowercased
    - stemmed with Porter stemmer
    """
    tokens = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        stem = stemmer.stem(token)
        tokens.append(stem)
    return tokens


# ---------- Index construction ----------

def traverse_json_files(root_dir):
    """Yield full paths to all .json files under root_dir."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)


def build_inverted_index(root_dir):
    """
    Build an in-memory inverted index over all HTML files under root_dir.

    Returns:
        index: dict(term -> dict(doc_id -> term_frequency))
        doc_id_map: dict(doc_id -> file_path)
    """
    index = defaultdict(dict)       # term -> {doc_id: tf}
    doc_id_map = {}                 # doc_id -> file_path
    next_doc_id = 0

    for file_path in traverse_json_files(root_dir):
        next_doc_id += 1
        doc_id = next_doc_id
        doc_id_map[doc_id] = file_path

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                page = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read/parse {file_path}: {e}", file=sys.stderr)
            continue

        # Adjust these keys JSON schema is slightly different.
        html = (
            page.get("content")
            or page.get("html")
            or ""
        )

        # For doc_id_map, identified by their URLs”)
        url = page.get("url", file_path)
        doc_id_map[doc_id] = url

        text = extract_visible_text(html)
        tokens = tokenize(text)

        # term frequency in this document
        tf = Counter(tokens)

        # update inverted index
        for term, freq in tf.items():
            index[term][doc_id] = freq

        if doc_id % 100 == 0:
            print(f"Indexed {doc_id} documents...", file=sys.stderr)

    return index, doc_id_map


# ---------- Serialization & analytics ----------

def save_json(obj, path):
    """
    Save obj as pretty-printed JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_index_size_kb(index_path, docmap_path):
    """
    Return total size on disk (KB) of the index + doc map files.
    """
    total_bytes = os.path.getsize(index_path) + os.path.getsize(docmap_path)
    return total_bytes / 1024.0


def main():
    if len(sys.argv) != 4:
        print("Usage: python build_index.py ROOT_DIR index.json doc_ids.json")
        sys.exit(1)

    root_dir = sys.argv[1]
    index_path = sys.argv[2]
    docmap_path = sys.argv[3]

    print(f"Building inverted index over HTML files under: {root_dir}")

    index, doc_id_map = build_inverted_index(root_dir)

    # Convert defaultdict to normal dict for JSON
    index_as_dict = {term: postings for term, postings in index.items()}

    print("Saving index and docID map to disk...")
    save_json(index_as_dict, index_path)
    save_json(doc_id_map, docmap_path)

    # ----- Analytics required for MS1 -----
    num_docs = len(doc_id_map)
    num_tokens = len(index_as_dict)
    size_kb = compute_index_size_kb(index_path, docmap_path)

    print("\n=== Milestone 1 Analytics ===")
    print(f"Number of indexed documents: {num_docs}")
    print(f"Number of unique tokens   : {num_tokens}")
    print(f"Total index size on disk  : {size_kb:.2f} KB")
    print("=============================")


if __name__ == "__main__":
    main()
