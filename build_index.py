#!/usr/bin/env python3
"""
build_index.py (Developer Flavor, with extra credit)

Usage:
  python3 build_index.py ROOT_DIR OUT_DIR

Outputs in OUT_DIR:
  docinfo.json            doc_id -> {"url":..., "norm":..., "norm2":..., "dup_urls":[...]}
  lexicon.json            unigram term -> [df, offset, nbytes]
  postings.bin            unigram postings (seekable)
  lexicon2.json           bigram term -> [df, offset, nbytes]
  postings2.bin           bigram postings (seekable)
  blocks_unigram/         partial blocks (proof of >=3 offloads)
  blocks_bigram/          partial blocks for 2-gram index
  canonical_map.json      original_doc_id -> canonical_doc_id (dups eliminated)
"""

import os, re, sys, json, math, heapq, hashlib, base64
from collections import Counter, defaultdict
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import warnings
from bs4 import XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
# ------------------ Tokenization / Stemming ------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
stemmer = PorterStemmer()

def tokenize(text: str):
    out = []
    if not text:
        return out
    for m in TOKEN_RE.finditer(text):
        out.append(stemmer.stem(m.group(0).lower()))
    return out

# ------------------ HTML Parsing ------------------
def soup_from_html(html: str):
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    return BeautifulSoup(html or "", "lxml")

def extract_zones(soup: BeautifulSoup):
    # remove non-visible conditions
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title_text = soup.title.get_text(" ") if soup.title else ""
    head_text = " ".join(t.get_text(" ") for t in soup.find_all(["h1", "h2", "h3"]))
    bold_text = " ".join(t.get_text(" ") for t in soup.find_all(["b", "strong"]))

    # body text: prefer soup.body if exists
    body_text = soup.body.get_text(" ") if soup.body else soup.get_text(" ")
    return body_text, title_text, head_text, bold_text

def extract_body_tokens_positions(body_text: str):
    """
    Return body_tokens (list), positions_map(term->list of positions)
    Positions are token index in body token sequence (0-based).
    """
    toks = tokenize(body_text)
    pos = defaultdict(list)
    for i, t in enumerate(toks):
        pos[t].append(i)
    return toks, pos

def extract_anchors(soup: BeautifulSoup, base_url: str):
    """
    Return list of (target_url, anchor_text)
    """
    results = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        if not href:
            continue
        # ignore non-http-ish
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        try:
            abs_url = urljoin(base_url, href) if base_url else href
        except Exception:
            continue

        # remove fragment
        abs_url, _ = urldefrag(abs_url)
        anchor_text = a.get_text(" ") or ""
        results.append((abs_url, anchor_text))
    return results

# ------------------ URL Normalization ------------------
def normalize_url(u: str):
    """
    Normalize URL enough for matching within corpus:
    - remove fragment (already done elsewhere)
    - lowercase scheme+host
    - strip default ports
    - keep path/query
    """
    if not u:
        return ""
    try:
        p = urlparse(u)
    except Exception:
        return u

    scheme = (p.scheme or "").lower()
    netloc = (p.netloc or "").lower()

    # strip default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # normalize path: keep as-is (ICS URLs can be case-sensitive on path sometimes)
    path = p.path or ""
    query = p.query or ""
    # drop params
    return urlunparse((scheme, netloc, path, "", query, ""))

# ------------------ Exact Duplicate (SHA256) ------------------
def exact_fingerprint(body_text: str):
    # collapse whitespace for stability
    norm = re.sub(r"\s+", " ", body_text or "").strip()
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()

# ------------------ Near Duplicate (SimHash + LSH) ------------------
def _hash64(s: str):
    # stable 64-bit hash from sha1
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def simhash64(tokens):
    """
    64-bit simhash of token multiset.
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
    return (a ^ b).bit_count()

# LSH: 4 bands of 16 bits
def bands(fp: int):
    return [(k, (fp >> (16 * k)) & 0xFFFF) for k in range(4)]

NEAR_DUP_HAMMING = 5  # <=5 considered near-duplicate (tune if needed)

# ------------------ Position Compression (delta + varint + base64) ------------------
def _varint_encode_number(x: int, out: bytearray):
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break

def _varint_decode_all(data: bytes):
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
    if not pos_list:
        return ""
    # delta encode
    deltas = [pos_list[0]]
    for i in range(1, len(pos_list)):
        deltas.append(pos_list[i] - pos_list[i - 1])
    buf = bytearray()
    for d in deltas:
        _varint_encode_number(d, buf)
    return base64.b64encode(bytes(buf)).decode("ascii")

def decode_positions(s):
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

# ------------------ Partial Block Flush / Merge ------------------
def flush_block_unigram(block_index, block_dir, block_id):
    """
    block_index: term -> {doc_id: (tfw, pos_b64)}
    write: term \t JSON([[doc, tfw, pos_b64], ...])
    """
    os.makedirs(block_dir, exist_ok=True)
    path = os.path.join(block_dir, f"block_{block_id:04d}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(block_index.keys()):
            postings = block_index[term]
            plist = [[doc, tfw, posb] for doc, (tfw, posb) in postings.items()]
            plist.sort(key=lambda x: x[0])
            f.write(term + "\t" + json.dumps(plist, separators=(",", ":")) + "\n")
    return path

def flush_block_bigram(block_index, block_dir, block_id):
    """
    block_index: bigram -> {doc_id: tf}
    write: bigram \t JSON([[doc, tf], ...])
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
        term, payload = line.rstrip("\n").split("\t", 1)
        self.term = term
        self.payload = json.loads(payload)

    def advance(self):
        if not self.eof:
            self._advance()

def merge_blocks_unigram(block_paths, out_postings, out_lexicon, N_docs):
    """
    merge unigram blocks -> postings.bin + lexicon.json
    doc norm computed with:
      idf = log10(N/df)
      w_td = (1 + log10(tfw)) * idf
      norm = sqrt(sum w_td^2)
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

            # consolidate duplicate doc ids
            consolidated = []
            last = None
            tfw_acc = 0
            posb = ""
            for doc, tfw, pb in merged:
                if doc == last:
                    tfw_acc += tfw
                    # positions from body should be identical; keep existing non-empty
                    if not posb and pb:
                        posb = pb
                else:
                    if last is not None:
                        consolidated.append([last, tfw_acc, posb])
                    last = doc
                    tfw_acc = tfw
                    posb = pb
            if last is not None:
                consolidated.append([last, tfw_acc, posb])

            df = len(consolidated)
            idf = math.log10(N_docs / df) if df > 0 else 0.0

            if idf != 0.0:
                for doc, tfw, _ in consolidated:
                    if tfw > 0:
                        w_td = (1.0 + math.log10(tfw)) * idf
                        norm_sq[doc] += w_td * w_td

            offset = outp.tell()
            rec = (term + "\t" + json.dumps(consolidated, separators=(",", ":")) + "\n").encode("utf-8")
            outp.write(rec)
            lexicon[term] = [df, offset, len(rec)]

    with open(out_lexicon, "w", encoding="utf-8") as f:
        json.dump(lexicon, f)

    return norm_sq

def merge_blocks_bigram(block_paths, out_postings, out_lexicon, N_docs):
    """
    merge bigram blocks -> postings2.bin + lexicon2.json
    compute norm2 similarly using tf bigram.
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

# ------------------ Main Pipeline ------------------
# weights for zones
W_TITLE, W_HEAD, W_BOLD, W_ANCHOR = 5, 3, 2, 2

# flush thresholds (tune)
FLUSH_EVERY_DOCS = 15000
MAX_POSTINGS_IN_MEM = 900_000  # counts unigram postings roughly

def traverse_json_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)

def read_page(path):
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

    # -------- Pass 1: assign doc_ids, collect urls, build url->doc mapping --------
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
        # if collision, keep first (duplicates handled later)
        url_to_doc.setdefault(nurl, i)

    # -------- Pass 2: detect duplicates + accumulate anchor words to targets --------
    exact_seen = {}  # sha -> canonical_doc
    lsh = defaultdict(list)  # (band_k, band_val) -> [canonical_doc,...]
    sim_fp = {}  # canonical_doc -> simhash

    canonical_of = {}     # doc -> canonical doc
    dup_urls = defaultdict(list)  # canonical -> [dup_url,...]

    anchor_tmp = defaultdict(Counter)  # target_original_doc -> Counter(tokens)

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

        # exact dup
        sha = exact_fingerprint(body_text)

        # near dup candidates
        body_tokens = tokenize(body_text)
        fp = simhash64(body_tokens)

        cand = None
        if sha in exact_seen:
            cand = exact_seen[sha]
        else:
            # LSH lookup for near dup
            candidates = set()
            for bk, bv in bands(fp):
                candidates.update(lsh.get((bk, bv), []))
            best = None
            best_dist = 10**9
            for cdoc in candidates:
                dist = hamming64(fp, sim_fp[cdoc])
                if dist < best_dist:
                    best_dist = dist
                    best = cdoc
            if best is not None and best_dist <= NEAR_DUP_HAMMING:
                cand = best

        if cand is None:
            # new canonical
            canonical_of[doc_id] = doc_id
            canonical_docs.add(doc_id)
            exact_seen[sha] = doc_id
            sim_fp[doc_id] = fp
            for bk, bv in bands(fp):
                lsh[(bk, bv)].append(doc_id)
        else:
            canonical_of[doc_id] = cand
            dup_urls[cand].append(base_url)

        # anchor extraction (store to target original doc; remap to canonical later)
        anchors = extract_anchors(soup, base_url)
        for tgt_url, atext in anchors:
            ntgt = normalize_url(tgt_url)
            tgt_doc = url_to_doc.get(ntgt)
            if not tgt_doc:
                continue
            toks = tokenize(atext)
            if toks:
                anchor_tmp[tgt_doc].update(toks)

    # remap anchors to canonical targets
    anchor_tf = defaultdict(Counter)  # canonical_doc -> Counter(tokens)
    for tgt_orig, cnt in anchor_tmp.items():
        can = canonical_of.get(tgt_orig, tgt_orig)
        anchor_tf[can].update(cnt)

    # -------- Pass 3: build disk index for canonical docs only (unigram+pos + bigram) --------
    blocks_uni_dir = os.path.join(out_dir, "blocks_unigram")
    blocks_bi_dir = os.path.join(out_dir, "blocks_bigram")
    unigram_block_index = defaultdict(dict)  # term -> {doc: (tfw, posb64)}
    bigram_block_index = defaultdict(dict)   # bigram -> {doc: tf}
    postings_in_mem = 0
    flush_count = 0
    bi_flush_count = 0
    block_paths_uni = []
    block_paths_bi = []

    # docinfo only for canonical docs (indexed docs)
    docinfo = {}

    # indexed-doc count N for idf
    indexed_docs_list = [d for d in range(1, len(paths) + 1) if canonical_of.get(d, d) == d]
    N_indexed = len(indexed_docs_list)

    for doc_id, path in enumerate(paths, start=1):
        if canonical_of.get(doc_id, doc_id) != doc_id:
            continue  # eliminate duplicates from index

        page = read_page(path)
        if not page:
            continue
        raw_url = page.get("url") or path
        raw_url, _ = urldefrag(raw_url)
        base_url = normalize_url(raw_url)

        html = page.get("content") or page.get("html") or ""
        soup = soup_from_html(html)
        body_text, title_text, head_text, bold_text = extract_zones(soup)

        body_tokens, pos_map = extract_body_tokens_positions(body_text)
        tf_body = Counter(body_tokens)
        tf_title = Counter(tokenize(title_text))
        tf_head = Counter(tokenize(head_text))
        tf_bold = Counter(tokenize(bold_text))
        tf_anchor = anchor_tf.get(doc_id, Counter())

        # weighted tfw
        tfw = Counter()
        tfw.update(tf_body)
        for t, c in tf_title.items():  tfw[t] += W_TITLE * c
        for t, c in tf_head.items():   tfw[t] += W_HEAD * c
        for t, c in tf_bold.items():   tfw[t] += W_BOLD * c
        for t, c in tf_anchor.items(): tfw[t] += W_ANCHOR * c

        # add unigram postings with compressed positions (positions from body only)
        for term, wtf in tfw.items():
            posb = encode_positions(pos_map.get(term, []))  # empty if not in body
            unigram_block_index[term][doc_id] = (wtf, posb)
        postings_in_mem += len(tfw)

        # build 2-gram index from body token order (bigrams)
        bi_tf = Counter()
        for i in range(len(body_tokens) - 1):
            bi = body_tokens[i] + " " + body_tokens[i + 1]
            bi_tf[bi] += 1
        for bi, tf in bi_tf.items():
            bigram_block_index[bi][doc_id] = tf

        # flush (unigram drives the required >=3 offloads)
        must_flush = (doc_id % FLUSH_EVERY_DOCS == 0) or (postings_in_mem >= MAX_POSTINGS_IN_MEM)
        if must_flush and unigram_block_index:
            flush_count += 1
            bp = flush_block_unigram(unigram_block_index, blocks_uni_dir, flush_count)
            block_paths_uni.append(bp)
            unigram_block_index.clear()
            postings_in_mem = 0
            print(f"[FLUSH] unigram {bp} (flush_count={flush_count})", file=sys.stderr)

            # keep bigram blocks in sync (optional flush)
            if bigram_block_index:
                bi_flush_count += 1
                bp2 = flush_block_bigram(bigram_block_index, blocks_bi_dir, bi_flush_count)
                block_paths_bi.append(bp2)
                bigram_block_index.clear()
                print(f"[FLUSH] bigram  {bp2} (flush_count={bi_flush_count})", file=sys.stderr)

        # docinfo placeholder; norms filled after merge
        docinfo[doc_id] = {"url": base_url, "norm": 1.0, "norm2": 1.0, "dup_urls": dup_urls.get(doc_id, [])}

    # final flush
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
        print(f"[WARN] unigram flush_count={flush_count} < 3. Lower thresholds!", file=sys.stderr)

    # merge to final postings + lexicon
    postings_path = os.path.join(out_dir, "postings.bin")
    lexicon_path = os.path.join(out_dir, "lexicon.json")
    norm_sq = merge_blocks_unigram(block_paths_uni, postings_path, lexicon_path, N_indexed)

    postings2_path = os.path.join(out_dir, "postings2.bin")
    lexicon2_path = os.path.join(out_dir, "lexicon2.json")
    norm2_sq = merge_blocks_bigram(block_paths_bi, postings2_path, lexicon2_path, N_indexed) if block_paths_bi else defaultdict(float)

    # write norms
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
    print(f"Exact+Near eliminated : {len(paths)-N_indexed}")
    print(f"Unigram flush blocks  : {flush_count} (must be >=3)")
    print(f"Bigram blocks         : {bi_flush_count}")
    print("Outputs:")
    print(f"  {postings_path}")
    print(f"  {lexicon_path}")
    print(f"  {postings2_path}")
    print(f"  {lexicon2_path}")
    print(f"  {os.path.join(out_dir,'docinfo.json')}")
    print("==================")

if __name__ == "__main__":
    main()
