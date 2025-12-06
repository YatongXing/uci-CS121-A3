#!/usr/bin/env python3
"""
build_index.py (Developer Flavor, with extra credit features)

Description:
  This script builds an inverted index from a corpus of JSON files containing web pages.
  It implements several advanced Information Retrieval concepts:
  1. Tokenization & Stemming (PorterStemmer).
  2. Zone Scoring: Weights terms differently based on location (Title, Headers, Bold, Anchors).
  3. Duplicate Detection:
     - Exact duplicates using SHA256.
     - Near-duplicates using SimHash and LSH (Locality Sensitive Hashing).
  4. Index Compression:
     - Delta Encoding for positions.
     - Variable Byte (VarInt) encoding for integers.
  5. SPIMI-style Indexing:
     - Writes partial blocks to disk when memory limit is reached.
     - Merges blocks using an N-way merge sort (Heapsort).
  6. 2-gram (Bigram) Indexing: Supports phrase search.

Outputs in OUT_DIR:
  docinfo.json            Metadata: doc_id -> {"url", "norm", "norm2", "dup_urls"}
  lexicon.json            Unigram Dictionary: term -> [df, offset, nbytes]
  postings.bin            Unigram Postings (Binary, seekable)
  lexicon2.json           Bigram Dictionary
  postings2.bin           Bigram Postings
  canonical_map.json      Mapping: original_doc_id -> canonical_doc_id
"""

import os, re, sys, json, math, heapq, hashlib, base64
from collections import Counter, defaultdict
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import warnings
from bs4 import XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning

# Tokenization and Stemming
# Regex to identify alphanumeric tokens
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
stemmer = PorterStemmer()


def tokenize(text: str):
    """
    Splits text into tokens, converts to lowercase, and applies Porter Stemming.
    """
    out = []
    if not text:
        return out
    for m in TOKEN_RE.finditer(text):
        out.append(stemmer.stem(m.group(0).lower()))
    return out


# HTML Parsing
def soup_from_html(html: str):
    """
    Safely creates a BeautifulSoup object, suppressing common XML/HTML warnings.
    """
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    return BeautifulSoup(html or "", "lxml")


def extract_zones(soup: BeautifulSoup):
    """
    Extracts text from specific HTML zones for weighted scoring.
    Returns: body_text, title_text, header_text (h1-h3), bold_text (b, strong)
    """
    # Remove non-visible tags to avoid indexing scripts as content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title_text = soup.title.get_text(" ") if soup.title else ""
    head_text = " ".join(t.get_text(" ") for t in soup.find_all(["h1", "h2", "h3"]))
    bold_text = " ".join(t.get_text(" ") for t in soup.find_all(["b", "strong"]))

    # Use soup.body if available, otherwise fallback to whole document text
    body_text = soup.body.get_text(" ") if soup.body else soup.get_text(" ")
    return body_text, title_text, head_text, bold_text


def extract_body_tokens_positions(body_text: str):
    """
    Tokenizes body text and records the position (index) of every token.
    Used for proximity search and position compression.
    Returns:
       - toks: list of tokens in order
       - pos: dict mapping term -> list of positions [0, 5, 12...]
    """
    toks = tokenize(body_text)
    pos = defaultdict(list)
    for i, t in enumerate(toks):
        pos[t].append(i)
    return toks, pos


def extract_anchors(soup: BeautifulSoup, base_url: str):
    """
    Extracts all <a> tags. Resolves relative URLs to absolute URLs.
    Returns list of (target_url, anchor_text).
    Note: Anchor text is used to index the TARGET page, not the source page.
    """
    results = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        if not href:
            continue
        # Ignore non-navigational links
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        try:
            abs_url = urljoin(base_url, href) if base_url else href
        except Exception:
            continue

        # Remove fragment (#section) for consistency
        abs_url, _ = urldefrag(abs_url)
        anchor_text = a.get_text(" ") or ""
        results.append((abs_url, anchor_text))
    return results


# URL Normalization
def normalize_url(u: str):
    """
    Standardizes URLs to ensure uniqueness.
    - Lowers case of scheme and host.
    - Removes default ports (80 for http, 443 for https).
    - Preserves path and query parameters.
    """
    if not u:
        return ""
    try:
        p = urlparse(u)
    except Exception:
        return u

    scheme = (p.scheme or "").lower()
    netloc = (p.netloc or "").lower()

    # Strip default ports to treat http://example.com:80 as http://example.com
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = p.path or ""
    query = p.query or ""
    return urlunparse((scheme, netloc, path, "", query, ""))


# Exact Duplicate (SHA256)
def exact_fingerprint(body_text: str):
    """
    Generates a SHA256 hash of the normalized body text.
    Used to detect 100% identical content.
    """
    norm = re.sub(r"\s+", " ", body_text or "").strip()
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()


# Near Duplicate (SimHash + LSH)
def _hash64(s: str):
    """Generates a 64-bit integer hash for a string."""
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def simhash64(tokens):
    """
    Calculates a 64-bit SimHash fingerprint for a list of tokens.
    SimHash Property: Similar documents have similar hash values (low Hamming distance).
    Algorithm:
    1. Initialize vector V of 64 zeros.
    2. For each token:
       - Hash it to 64 bits.
       - For bit i: if 1, add weight to V[i]; if 0, subtract weight.
    3. Final Fingerprint: if V[i] >= 0, bit is 1, else 0.
    """
    if not tokens:
        return 0
    tf = Counter(tokens)
    v = [0] * 64
    for t, w in tf.items():
        h = _hash64(t)
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += w if bit else -w
    fp = 0
    for i in range(64):
        if v[i] >= 0:
            fp |= (1 << i)
    return fp


def hamming64(a: int, b: int):
    """Calculates number of differing bits between two 64-bit integers."""
    return (a ^ b).bit_count()


# Locality Sensitive Hashing (LSH) Configuration
# Split 64 bits into 4 bands of 16 bits.
# If two docs share a band (exact match of 16 bits), they are candidates for checking.
def bands(fp: int):
    return [(k, (fp >> (16 * k)) & 0xFFFF) for k in range(4)]


NEAR_DUP_HAMMING = 5  # Threshold: if diff <= 5 bits, consider as duplicate


# Position Compression (delta + varint + base64)
def _varint_encode_number(x: int, out: bytearray):
    """
    Encodes an integer into Variable Byte format.
    Uses 7 bits for data, 1 bit (MSB) as continuation flag.
    Efficient for small numbers (like delta gaps).
    """
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)  # Set continuation bit
        else:
            out.append(b)
            break


def _varint_decode_all(data: bytes):
    """Decodes a stream of bytes back into a list of integers."""
    nums = []
    x = 0
    shift = 0
    for b in data:
        x |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            nums.append(x)
            x = 0
            shift = 0
    return nums


def encode_positions(pos_list):
    """
    Compresses a list of positions:
    1. Calculate Deltas: [10, 15, 20] -> [10, 5, 5]
    2. VarInt Encode: Convert deltas to bytes.
    3. Base64: Encode bytes to string for JSON storage efficiency.
    """
    if not pos_list:
        return ""
    # Delta encoding
    deltas = [pos_list[0]]
    for i in range(1, len(pos_list)):
        deltas.append(pos_list[i] - pos_list[i - 1])
    buf = bytearray()
    for d in deltas:
        _varint_encode_number(d, buf)
    return base64.b64encode(bytes(buf)).decode("ascii")


def decode_positions(s):
    """Decompresses base64 encoded delta-varint string back to position list."""
    if not s:
        return []
    data = base64.b64decode(s.encode("ascii"))
    deltas = _varint_decode_all(data)
    pos = []
    cur = 0
    for i, d in enumerate(deltas):
        cur = d if i == 0 else cur + d
        pos.append(cur)
    return pos


# Partial Block Flush and Merge (SPIMI Logic)
# This section implements the Single-Pass In-Memory Indexing (SPIMI) strategy.
# It defines how to:
# 1. Flush in-memory data to disk when memory limits are reached.
# 2. Stream data back from disk.
# 3. Merge multiple temporary blocks into the final inverted index.

def flush_block_unigram(block_index, block_dir, block_id):
    """
    Writes the in-memory inverted index (block) to a TSV file on disk.
    The format written is:
    term [TAB] JSON([[doc_id, tfw, encoded_positions], ...])
    This ensures we clear RAM to process the next batch of documents.
    """
    os.makedirs(block_dir, exist_ok=True)
    path = os.path.join(block_dir, f"block_{block_id:04d}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        # Sort terms alphabetically to ensure the block is sorted
        for term in sorted(block_index.keys()):
            postings = block_index[term]
            # Convert dict {doc: data} to list [[doc, data], ...] and sort by doc_id
            plist = [[doc, tfw, posb] for doc, (tfw, posb) in postings.items()]
            plist.sort(key=lambda x: x[0])
            f.write(term + "\t" + json.dumps(plist, separators=(",", ":")) + "\n")
    return path


def flush_block_bigram(block_index, block_dir, block_id):
    """
    Writes the in-memory bigram index to disk.
    Similar to unigram flush, but bigrams usually do not store positions
    to save disk space, storing only the Term Frequency (TF).
    """
    os.makedirs(block_dir, exist_ok=True)
    path = os.path.join(block_dir, f"block_{block_id:04d}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(block_index.keys()):
            postings = block_index[term]
            plist = [[doc, tf] for doc, tf in postings.items()]
            plist.sort(key=lambda x: x[0])
            f.write(term + "\t" + json.dumps(plist, separators=(",", ":")) + "\n")
    return path


class BlockStream:
    """
    Helper class to read flushed blocks line-by-line during the merge phase.
    It buffers the file reading so we don't load the entire block back into RAM.
    """

    def __init__(self, path):
        self.fp = open(path, "r", encoding="utf-8")
        self.term = None
        self.payload = None
        self.eof = False
        self._advance()

    def _advance(self):
        line = self.fp.readline()
        if not line:
            self.eof = True
            self.term = None
            self.payload = None
            self.fp.close()
            return
        # Parse TSV line: term on left, JSON payload on right
        term, payload = line.rstrip("\n").split("\t", 1)
        self.term = term
        self.payload = json.loads(payload)

    def advance(self):
        if not self.eof:
            self._advance()


def merge_blocks_unigram(block_paths, out_postings, out_lexicon, N_docs):
    """
    Merges multiple partial index blocks into one final inverted index using N-way merge.
    This function also calculates the TF-IDF vector norms for document ranking.
    Outputs:
    1. postings.bin: A binary-like text file containing the merged lists.
    2. lexicon.json: A dictionary mapping 'term' -> [doc_freq, byte_offset, byte_length].
    """
    streams = [BlockStream(p) for p in block_paths]
    heap = []

    # Initialize Heap with the first term from each block stream
    for i, s in enumerate(streams):
        if not s.eof:
            heapq.heappush(heap, (s.term, i))

    lexicon = {}
    norm_sq = defaultdict(float)  # Accumulator for doc vector norms (sum of squares)

    with open(out_postings, "wb") as outp:
        while heap:
            # Pop the smallest term (alphabetically first) from the heap
            term, i = heapq.heappop(heap)
            merged = list(streams[i].payload)

            # Advance that stream to the next term
            streams[i].advance()
            if not streams[i].eof:
                heapq.heappush(heap, (streams[i].term, i))

            # Check if other blocks have the same term and merge them
            while heap and heap[0][0] == term:
                _, j = heapq.heappop(heap)
                merged.extend(streams[j].payload)
                streams[j].advance()
                if not streams[j].eof:
                    heapq.heappush(heap, (streams[j].term, j))

            # Sort the combined postings list by Document ID
            merged.sort(key=lambda x: x[0])

            # Consolidate duplicate doc IDs if any (safety check)
            consolidated = []
            last = None
            tfw_acc = 0
            posb = ""
            for doc, tfw, pb in merged:
                if doc == last:
                    tfw_acc += tfw
                    if not posb and pb: posb = pb
                else:
                    if last is not None:
                        consolidated.append([last, tfw_acc, posb])
                    last = doc
                    tfw_acc = tfw
                    posb = pb
            if last is not None:
                consolidated.append([last, tfw_acc, posb])

            # Calculate Document Frequency (df) and IDF
            df = len(consolidated)
            idf = math.log10(N_docs / df) if df > 0 else 0.0

            # Accumulate vector weights for Cosine Similarity normalization
            # Formula: W_td = (1 + log10(tf)) * idf
            if idf != 0.0:
                for doc, tfw, _ in consolidated:
                    if tfw > 0:
                        w_td = (1.0 + math.log10(tfw)) * idf
                        norm_sq[doc] += w_td * w_td

            # Write the final merged posting list to the postings file
            offset = outp.tell()
            rec = (term + "\t" + json.dumps(consolidated, separators=(",", ":")) + "\n").encode("utf-8")
            outp.write(rec)

            # Update Lexicon with offset and length for random access later
            lexicon[term] = [df, offset, len(rec)]

    # Save the lexicon to disk
    with open(out_lexicon, "w", encoding="utf-8") as f:
        json.dump(lexicon, f)

    return norm_sq


def merge_blocks_bigram(block_paths, out_postings, out_lexicon, N_docs):
    """
    Merges bigram blocks.
    This follows the exact same logic as the unigram merge, but tailored for
    bigram data structures (which usually lack position data).
    It also calculates a separate norm (norm2) for bigram-based ranking.
    """
    streams = [BlockStream(p) for p in block_paths]
    heap = []
    for i, s in enumerate(streams):
        if not s.eof:
            heapq.heappush(heap, (s.term, i))

    lexicon = {}
    norm_sq = defaultdict(float)

    with open(out_postings, "wb") as outp:
        while heap:
            term, i = heapq.heappop(heap)
            merged = list(streams[i].payload)

            streams[i].advance()
            if not streams[i].eof:
                heapq.heappush(heap, (streams[i].term, i))

            while heap and heap[0][0] == term:
                _, j = heapq.heappop(heap)
                merged.extend(streams[j].payload)
                streams[j].advance()
                if not streams[j].eof:
                    heapq.heappush(heap, (streams[j].term, j))

            merged.sort(key=lambda x: x[0])

            consolidated = []
            last = None
            acc = 0
            for doc, tf in merged:
                if doc == last:
                    acc += tf
                else:
                    if last is not None:
                        consolidated.append([last, acc])
                    last = doc
                    acc = tf
            if last is not None:
                consolidated.append([last, acc])

            df = len(consolidated)
            idf = math.log10(N_docs / df) if df > 0 else 0.0

            if idf != 0.0:
                for doc, tf in consolidated:
                    if tf > 0:
                        w_td = (1.0 + math.log10(tf)) * idf
                        norm_sq[doc] += w_td * w_td

            offset = outp.tell()
            rec = (term + "\t" + json.dumps(consolidated, separators=(",", ":")) + "\n").encode("utf-8")
            outp.write(rec)
            lexicon[term] = [df, offset, len(rec)]

    with open(out_lexicon, "w", encoding="utf-8") as f:
        json.dump(lexicon, f)

    return norm_sq


# Main Pipeline
# Zone Weights: Title is most important, followed by Headers, etc.
W_TITLE, W_HEAD, W_BOLD, W_ANCHOR = 5, 3, 2, 2

# Memory Management Thresholds
# If FLUSH_EVERY_DOCS is hit or Posting count exceeds MAX, we write to disk.
# This ensures we don't crash on large corpora (SPIMI constraint).
FLUSH_EVERY_DOCS = 15000
MAX_POSTINGS_IN_MEM = 900_000


def traverse_json_files(root_dir):
    """Recursively yields paths to all .json files in root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)


def read_page(path):
    """Helper to read JSON file safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_index.py ROOT_DIR OUT_DIR")
        sys.exit(1)

    root_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # PASS 1: Assign Doc IDs and build URL Map
    # Goal: Scan all files to create a consistent ID -> URL mapping.
    # We resolve relative URLs later, so we need to know the base URL of every doc.
    paths = []
    urls = {}
    url_to_doc = {}
    for p in traverse_json_files(root_dir):
        paths.append(p)

    for i, path in enumerate(paths, start=1):
        page = read_page(path)
        if not page:
            continue
        url = page.get("url") or path
        url, _ = urldefrag(url)
        nurl = normalize_url(url)
        urls[i] = nurl
        # If multiple files map to the same URL, the first one seen claims the ID.
        url_to_doc.setdefault(nurl, i)

    # PASS 2: Duplicate Detection & Anchor Text Aggregation
    # Goal:
    # 1. Identify Exact duplicates (SHA256) and Near duplicates (SimHash).
    # 2. Extract Anchor Text. Anchor text is special because it describes the
    #    Target page, not the Source page. We must aggregate it here to add to
    #    the target's index in Pass 3.

    exact_seen = {}  # sha256 -> canonical_doc_id
    lsh = defaultdict(list)  # (band_index, band_hash) -> [canonical_doc_ids]
    sim_fp = {}  # canonical_doc_id -> simhash_value

    canonical_of = {}  # Mapping: any_doc_id -> canonical_doc_id
    dup_urls = defaultdict(list)  # canonical_doc -> [list of duplicate URLs]

    anchor_tmp = defaultdict(Counter)  # target_original_doc_id -> Counter(tokens)
    canonical_docs = set()

    for doc_id, path in enumerate(paths, start=1):
        page = read_page(path)
        if not page:
            continue
        raw_url = page.get("url") or path
        raw_url, _ = urldefrag(raw_url)
        base_url = normalize_url(raw_url)

        html = page.get("content") or page.get("html") or ""
        soup = soup_from_html(html)
        body_text, _, _, _ = extract_zones(soup)

        # --- Duplicate Detection Logic ---
        sha = exact_fingerprint(body_text)
        body_tokens = tokenize(body_text)
        fp = simhash64(body_tokens)

        cand = None

        # 1. Check Exact Duplicate
        if sha in exact_seen:
            cand = exact_seen[sha]
        else:
            # 2. Check Near Duplicate using LSH
            candidates = set()
            for bk, bv in bands(fp):
                candidates.update(lsh.get((bk, bv), []))

            # Find closest match among candidates
            best = None
            best_dist = 10 ** 9
            for cdoc in candidates:
                dist = hamming64(fp, sim_fp[cdoc])
                if dist < best_dist:
                    best_dist = dist
                    best = cdoc

            # Verify if best candidate is actually close enough
            if best is not None and best_dist <= NEAR_DUP_HAMMING:
                cand = best

        # Decision: Is this a new canonical doc or a duplicate?
        if cand is None:
            # New canonical document found
            canonical_of[doc_id] = doc_id
            canonical_docs.add(doc_id)
            exact_seen[sha] = doc_id
            sim_fp[doc_id] = fp
            # Add to LSH buckets
            for bk, bv in bands(fp):
                lsh[(bk, bv)].append(doc_id)
        else:
            # Mark as duplicate of 'cand'
            canonical_of[doc_id] = cand
            dup_urls[cand].append(base_url)

        # --- Anchor Extraction ---
        anchors = extract_anchors(soup, base_url)
        for tgt_url, atext in anchors:
            ntgt = normalize_url(tgt_url)
            tgt_doc = url_to_doc.get(ntgt)
            # Only store anchor text if we know the target document exists in corpus
            if not tgt_doc:
                continue
            toks = tokenize(atext)
            if toks:
                anchor_tmp[tgt_doc].update(toks)

    # Remap anchor text from original target IDs to Canonical target IDs
    # (Because the target page might have been a duplicate, we need to point to the canonical one)
    anchor_tf = defaultdict(Counter)
    for tgt_orig, cnt in anchor_tmp.items():
        can = canonical_of.get(tgt_orig, tgt_orig)
        anchor_tf[can].update(cnt)

    # PASS 3: Index Construction (Canonical Docs Only)
    # Goal: Parse documents one last time to build the index.
    # 1. Calculate weighted TF (tfw) based on zones + incoming anchor text.
    # 2. Encode positions.
    # 3. Create Bigrams.
    # 4. Flush to disk periodically to manage memory.

    blocks_uni_dir = os.path.join(out_dir, "blocks_unigram")
    blocks_bi_dir = os.path.join(out_dir, "blocks_bigram")
    unigram_block_index = defaultdict(dict)  # term -> {doc: (tfw, posb64)}
    bigram_block_index = defaultdict(dict)  # bigram -> {doc: tf}
    postings_in_mem = 0
    flush_count = 0
    bi_flush_count = 0
    block_paths_uni = []
    block_paths_bi = []

    docinfo = {}

    # Identify set of valid docs for IDF calculation
    indexed_docs_list = [d for d in range(1, len(paths) + 1) if canonical_of.get(d, d) == d]
    N_indexed = len(indexed_docs_list)

    for doc_id, path in enumerate(paths, start=1):
        # Skip if this document was marked as a duplicate in Pass 2
        if canonical_of.get(doc_id, doc_id) != doc_id:
            continue

        page = read_page(path)
        if not page:
            continue

        # Re-parse HTML
        raw_url = page.get("url") or path
        raw_url, _ = urldefrag(raw_url)
        base_url = normalize_url(raw_url)
        html = page.get("content") or page.get("html") or ""
        soup = soup_from_html(html)
        body_text, title_text, head_text, bold_text = extract_zones(soup)

        # Get Term Frequencies (TF) for all zones
        body_tokens, pos_map = extract_body_tokens_positions(body_text)
        tf_body = Counter(body_tokens)
        tf_title = Counter(tokenize(title_text))
        tf_head = Counter(tokenize(head_text))
        tf_bold = Counter(tokenize(bold_text))

        # Get Anchor Text TF (calculated in Pass 2)
        tf_anchor = anchor_tf.get(doc_id, Counter())

        # Calculate Combined Weighted TF (tfw)
        tfw = Counter()
        tfw.update(tf_body)  # Body weight = 1 (implied)
        for t, c in tf_title.items():  tfw[t] += W_TITLE * c
        for t, c in tf_head.items():   tfw[t] += W_HEAD * c
        for t, c in tf_bold.items():   tfw[t] += W_BOLD * c
        for t, c in tf_anchor.items(): tfw[t] += W_ANCHOR * c

        # --- Update Unigram Index ---
        for term, wtf in tfw.items():
            # Get positions from body only (for phrase search accuracy)
            posb = encode_positions(pos_map.get(term, []))
            unigram_block_index[term][doc_id] = (wtf, posb)
        postings_in_mem += len(tfw)

        # --- Update Bigram Index ---
        # Generate bigrams from body token sequence
        bi_tf = Counter()
        for i in range(len(body_tokens) - 1):
            bi = body_tokens[i] + " " + body_tokens[i + 1]
            bi_tf[bi] += 1
        for bi, tf in bi_tf.items():
            bigram_block_index[bi][doc_id] = tf

        # --- Check Flush Condition ---
        # If we have too many postings in RAM or processed enough docs, flush to disk.
        must_flush = (doc_id % FLUSH_EVERY_DOCS == 0) or (postings_in_mem >= MAX_POSTINGS_IN_MEM)
        if must_flush and unigram_block_index:
            flush_count += 1
            bp = flush_block_unigram(unigram_block_index, blocks_uni_dir, flush_count)
            block_paths_uni.append(bp)
            unigram_block_index.clear()
            postings_in_mem = 0
            print(f"[FLUSH] unigram {bp} (flush_count={flush_count})", file=sys.stderr)

            # Flush bigrams as well to keep sync
            if bigram_block_index:
                bi_flush_count += 1
                bp2 = flush_block_bigram(bigram_block_index, blocks_bi_dir, bi_flush_count)
                block_paths_bi.append(bp2)
                bigram_block_index.clear()
                print(f"[FLUSH] bigram  {bp2} (flush_count={bi_flush_count})", file=sys.stderr)

        # Initialize DocInfo (Norms will be updated after merge)
        docinfo[doc_id] = {
            "url": base_url,
            "norm": 1.0,
            "norm2": 1.0,
            "dup_urls": dup_urls.get(doc_id, [])
        }

    # --- Final Flush for Remaining Data ---
    if unigram_block_index:
        flush_count += 1
        bp = flush_block_unigram(unigram_block_index, blocks_uni_dir, flush_count)
        block_paths_uni.append(bp)
        print(f"[FLUSH] unigram {bp} (flush_count={flush_count})", file=sys.stderr)

    if bigram_block_index:
        bi_flush_count += 1
        bp2 = flush_block_bigram(bigram_block_index, blocks_bi_dir, bi_flush_count)
        block_paths_bi.append(bp2)
        print(f"[FLUSH] bigram  {bp2} (flush_count={bi_flush_count})", file=sys.stderr)

    if flush_count < 3:
        print(f"[WARN] unigram flush_count={flush_count} < 3. Adjust thresholds for demonstration!", file=sys.stderr)

    # MERGE: Combine partial blocks into final index
    postings_path = os.path.join(out_dir, "postings.bin")
    lexicon_path = os.path.join(out_dir, "lexicon.json")
    # merge_blocks returns the sum of squared weights for every doc
    norm_sq = merge_blocks_unigram(block_paths_uni, postings_path, lexicon_path, N_indexed)

    postings2_path = os.path.join(out_dir, "postings2.bin")
    lexicon2_path = os.path.join(out_dir, "lexicon2.json")
    norm2_sq = merge_blocks_bigram(block_paths_bi, postings2_path, lexicon2_path,
                                   N_indexed) if block_paths_bi else defaultdict(float)

    # Finalize norms (square root) and write docinfo
    for d in docinfo:
        v1 = norm_sq.get(d, 0.0)
        v2 = norm2_sq.get(d, 0.0)
        docinfo[d]["norm"] = math.sqrt(v1) if v1 > 0 else 1.0
        docinfo[d]["norm2"] = math.sqrt(v2) if v2 > 0 else 1.0

    with open(os.path.join(out_dir, "docinfo.json"), "w", encoding="utf-8") as f:
        json.dump(docinfo, f)

    with open(os.path.join(out_dir, "canonical_map.json"), "w", encoding="utf-8") as f:
        json.dump(canonical_of, f)

    print("\n=== Build Done ===")
    print(f"All docs scanned      : {len(paths)}")
    print(f"Indexed (canonical)   : {N_indexed}")
    print(f"Exact+Near eliminated : {len(paths) - N_indexed}")
    print(f"Unigram flush blocks  : {flush_count}")
    print(f"Bigram blocks         : {bi_flush_count}")
    print("Outputs:")
    print(f"  {postings_path}")
    print(f"  {lexicon_path}")
    print(f"  {postings2_path}")
    print(f"  {lexicon2_path}")
    print(f"  {os.path.join(out_dir, 'docinfo.json')}")
    print("==================")


if __name__ == "__main__":
    main()