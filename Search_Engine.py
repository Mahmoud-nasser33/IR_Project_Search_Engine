# search_engine_enhanced.py
import json
import math
import os
import pickle
import re
import threading
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import nltk
import numpy as np
import pandas as pd
import requests
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gensim.models import Word2Vec
from langdetect import DetectorFactory
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("KAGGLEHUB_CACHE", CACHE_DIR)

CORPUS_CSV = os.path.join(CACHE_DIR, "corpus.csv")
INDEX_PKL = os.path.join(CACHE_DIR, "inverted_index.pkl")
NORMS_PKL = os.path.join(CACHE_DIR, "doc_norms.pkl")
IDF_PKL = os.path.join(CACHE_DIR, "idf.pkl")
W2V_PATH = os.path.join(CACHE_DIR, "word2vec.model")
BERT_PATH = os.path.join(CACHE_DIR, "bert_embeddings.npy")
COVERS_JSON = os.path.join(CACHE_DIR, "covers.json")

RANDOM_SEED = 42

for pkg, path in [("punkt", "tokenizers/punkt"),
                  ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


def preprocess(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return tokens


def load_corpus():
    if os.path.exists(CORPUS_CSV):
        print(f"[corpus] loading cached corpus from {CORPUS_CSV}")
        df = pd.read_csv(CORPUS_CSV)
        df["genre_list"] = df["genre_list"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else []
        )
        return df

    print("[corpus] downloading CMU book summaries from Kaggle...")
    import kagglehub
    path = kagglehub.dataset_download("ymaricar/cmu-book-summary-dataset")
    csv_file = os.path.join(path, "booksummaries.txt")

    raw = pd.read_csv(
        csv_file, sep="\t", header=None,
        names=["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"],
        on_bad_lines="skip",
    )

    df = raw[["title", "author", "pub_date", "genres", "summary"]].dropna(subset=["summary", "title"])
    df = df[df["summary"].str.len() > 100].reset_index(drop=True)
    df.insert(0, "doc_id", range(1, len(df) + 1))

    def parse_genres(g):
        if not isinstance(g, str) or not g:
            return []
        try:
            return list(json.loads(g).values())
        except Exception:
            return []

    df["genre_list"] = df["genres"].apply(parse_genres)

    out = df.copy()
    out["genre_list"] = out["genre_list"].apply(json.dumps)
    out.to_csv(CORPUS_CSV, index=False)
    print(f"[corpus] built and cached {len(df):,} documents")
    return df


print("=" * 70)
print("Enhanced Book Search Engine - Initialising")
print("=" * 70)

df = load_corpus()
print(f"[corpus] {len(df):,} books")

print("[preprocess] tokenising corpus...")
df["tokens"] = df["summary"].apply(preprocess)

DOC_INDEX = {int(row["doc_id"]): row for _, row in df.iterrows()}


def build_inverted_index():
    if os.path.exists(INDEX_PKL):
        print(f"[index] loading cached inverted index from {INDEX_PKL}")
        with open(INDEX_PKL, "rb") as f:
            return pickle.load(f)

    print("[index] building inverted index...")
    inv = defaultdict(dict)
    for _, row in df.iterrows():
        doc_id = int(row["doc_id"])
        tf = defaultdict(int)
        for t in row["tokens"]:
            tf[t] += 1
        for term, freq in tf.items():
            inv[term][doc_id] = freq
    inv = dict(inv)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(inv, f)
    print(f"[index] cached {len(inv):,} terms")
    return inv


inverted = build_inverted_index()


def build_tfidf():
    if os.path.exists(IDF_PKL) and os.path.exists(NORMS_PKL):
        print("[tfidf] loading cached IDF and document norms")
        with open(IDF_PKL, "rb") as f:
            _idf = pickle.load(f)
        with open(NORMS_PKL, "rb") as f:
            _norms = pickle.load(f)
        return _idf, _norms

    print("[tfidf] computing IDF and document norms...")
    N = len(df)
    df_dict = {term: len(postings) for term, postings in inverted.items()}
    _idf = {term: math.log(N / d) for term, d in df_dict.items()}

    def w(tf, term):
        if tf <= 0 or term not in _idf:
            return 0.0
        return (1 + math.log(tf)) * _idf[term]

    _norms = {}
    for _, row in df.iterrows():
        c = Counter(row["tokens"])
        s = sum(w(tf, t) ** 2 for t, tf in c.items())
        _norms[int(row["doc_id"])] = math.sqrt(s) if s > 0 else 1.0

    with open(IDF_PKL, "wb") as f:
        pickle.dump(_idf, f)
    with open(NORMS_PKL, "wb") as f:
        pickle.dump(_norms, f)
    return _idf, _norms


idf, doc_norms = build_tfidf()


def tfidf_weight(tf, term):
    if tf <= 0 or term not in idf:
        return 0.0
    return (1 + math.log(tf)) * idf[term]


def rank_tfidf(q_tokens, top_k=20):
    if not q_tokens:
        return []
    q_tf = Counter(q_tokens)
    q_w = {t: tfidf_weight(tf, t) for t, tf in q_tf.items() if t in idf}
    q_norm = math.sqrt(sum(v * v for v in q_w.values()))
    if q_norm == 0:
        return []
    scores = defaultdict(float)
    for term, qw in q_w.items():
        for doc_id, tf in inverted.get(term, {}).items():
            scores[doc_id] += qw * tfidf_weight(tf, term)
    ranked = [(d, dot / (q_norm * doc_norms[d])) for d, dot in scores.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rocchio_expand(q_tokens, fb=5, n_expand=8, alpha=1.0, beta=0.75):
    initial = rank_tfidf(q_tokens, top_k=fb)
    if not initial:
        return q_tokens
    centroid = defaultdict(float)
    for doc_id, _ in initial:
        row = DOC_INDEX[doc_id]
        c = Counter(row["tokens"])
        for t, tf in c.items():
            centroid[t] += tfidf_weight(tf, t)
    for t in centroid:
        centroid[t] /= len(initial)
    qtf = Counter(q_tokens)
    combo = defaultdict(float)
    for t, tf in qtf.items():
        combo[t] += alpha * tfidf_weight(tf, t)
    for t, w in centroid.items():
        combo[t] += beta * w
    orig = set(q_tokens)
    cands = sorted(((t, w) for t, w in combo.items() if t not in orig),
                   key=lambda x: x[1], reverse=True)
    return q_tokens + [t for t, _ in cands[:n_expand]]


def train_or_load_w2v():
    if os.path.exists(W2V_PATH):
        print(f"[w2v] loading cached Word2Vec model from {W2V_PATH}")
        return Word2Vec.load(W2V_PATH)
    print("[w2v] training Word2Vec on corpus (this may take a few minutes)...")
    model = Word2Vec(
        sentences=df["tokens"].tolist(),
        vector_size=100, window=5, min_count=3, sg=1,
        workers=4, epochs=10, seed=RANDOM_SEED,
    )
    model.save(W2V_PATH)
    return model


w2v = train_or_load_w2v()


def w2v_expand(q_tokens, neighbours=3, sim_threshold=0.55):
    out, seen = list(q_tokens), set(q_tokens)
    for t in q_tokens:
        if t not in w2v.wv:
            continue
        for nbr, sim in w2v.wv.most_similar(t, topn=neighbours):
            if sim >= sim_threshold and nbr not in seen and nbr in inverted:
                out.append(nbr)
                seen.add(nbr)
    return out


bert_model = None
bert_emb = None
try:
    from sentence_transformers import SentenceTransformer

    print("[bert] loading sentence transformer (all-MiniLM-L6-v2)...")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(BERT_PATH):
        print(f"[bert] loading cached embeddings from {BERT_PATH}")
        bert_emb = np.load(BERT_PATH)
    else:
        print("[bert] encoding documents (one-time, may take several minutes)...")
        texts = df["summary"].astype(str).str.slice(0, 1500).tolist()
        bert_emb = bert_model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        np.save(BERT_PATH, bert_emb)
    print(f"[bert] embeddings ready, shape={bert_emb.shape}")
except Exception as e:
    print(f"[bert] disabled: {e}")
    bert_model = None
    bert_emb = None


def bert_rerank(q_text, candidates, top_k=20):
    if not candidates or bert_model is None or bert_emb is None:
        return candidates[:top_k]
    q = bert_model.encode(q_text, normalize_embeddings=True)
    ids = [d for d, _ in candidates]
    idx = [d - 1 for d in ids]
    sims = bert_emb[idx] @ q
    return sorted(zip(ids, sims.tolist()), key=lambda x: x[1], reverse=True)[:top_k]


def find_similar(doc_id, top_k=10):
    if bert_emb is None:
        return []
    if doc_id < 1 or doc_id > len(bert_emb):
        return []
    q = bert_emb[doc_id - 1]
    sims = bert_emb @ q
    pairs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    out = [(int(i + 1), float(s)) for i, s in pairs if (i + 1) != doc_id][:top_k]
    return out


DetectorFactory.seed = 0


def is_arabic_text(s):
    return any("؀" <= c <= "ۿ" for c in s)


def maybe_translate(text):
    if not text.strip():
        return text, None, "en"
    if not is_arabic_text(text):
        return text, None, "en"
    try:
        en = GoogleTranslator(source="ar", target="en").translate(text)
        if en and en.strip():
            return en, text, "ar"
    except Exception:
        pass
    return text, None, "ar"


COVERS_LOCK = threading.Lock()
if os.path.exists(COVERS_JSON):
    try:
        with open(COVERS_JSON, "r", encoding="utf-8") as f:
            covers_cache = json.load(f)
    except Exception:
        covers_cache = {}
else:
    covers_cache = {}
print(f"[covers] cache loaded with {len(covers_cache):,} entries")


def cover_key(title, author):
    return f"{(title or '').strip().lower()}|{(author or '').strip().lower()}"


def fetch_cover(title, author):
    k = cover_key(title, author)
    with COVERS_LOCK:
        if k in covers_cache:
            return covers_cache[k]
    try:
        params = {"title": title, "limit": 1}
        if author and author.lower() != "unknown":
            params["author"] = author
        r = requests.get("https://openlibrary.org/search.json", params=params, timeout=4)
        if r.status_code != 200:
            url = None
        else:
            docs = (r.json() or {}).get("docs") or []
            cid = docs[0].get("cover_i") if docs else None
            url = f"https://covers.openlibrary.org/b/id/{cid}-M.jpg" if cid else None
    except Exception:
        url = None
    with COVERS_LOCK:
        covers_cache[k] = url
    return url


def enrich_covers(items):
    if not items:
        return items
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_cover, it["title"], it.get("author", "")): it for it in items}
        for f in futs:
            try:
                futs[f]["cover_url"] = f.result()
            except Exception:
                futs[f]["cover_url"] = None
    with COVERS_LOCK:
        try:
            tmp = COVERS_JSON + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(covers_cache, fp)
            os.replace(tmp, COVERS_JSON)
        except Exception:
            pass
    return items


def apply_filters(ranked, genres=None, authors=None, year_min=None, year_max=None):
    if not (genres or authors or year_min or year_max):
        return ranked
    gset = set(g.lower() for g in (genres or []))
    aset = set(a.lower() for a in (authors or []))
    out = []
    for doc_id, score in ranked:
        row = DOC_INDEX[doc_id]
        if gset:
            row_gs = set(str(g).lower() for g in (row["genre_list"] or []))
            if not (gset & row_gs):
                continue
        if aset and str(row["author"]).lower() not in aset:
            continue
        if year_min or year_max:
            yr = parse_year(row.get("pub_date", ""))
            if yr is None:
                continue
            if year_min and yr < year_min:
                continue
            if year_max and yr > year_max:
                continue
        out.append((doc_id, score))
    return out


def parse_year(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"(\d{4})", s)
    if not m:
        return None
    try:
        y = int(m.group(1))
        if 1000 <= y <= 2100:
            return y
    except Exception:
        pass
    return None


def make_items(ranked):
    items = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        row = DOC_INDEX[doc_id]
        summary = str(row["summary"])
        snippet = summary[:320].replace("\n", " ").strip()
        if len(summary) > 320:
            snippet += "..."
        items.append({
            "rank": rank,
            "doc_id": int(doc_id),
            "score": float(score),
            "title": str(row["title"]),
            "author": str(row["author"]) if pd.notna(row["author"]) else "Unknown",
            "year": parse_year(str(row.get("pub_date", ""))),
            "genres": list(row["genre_list"]) if isinstance(row["genre_list"], list) else [],
            "snippet": snippet,
            "cover_url": None,
        })
    enrich_covers(items)
    return items


def sort_items(items, sort_by):
    if sort_by == "title":
        items.sort(key=lambda x: x["title"].lower())
    elif sort_by == "author":
        items.sort(key=lambda x: x["author"].lower())
    elif sort_by == "year_desc":
        items.sort(key=lambda x: x["year"] or -1, reverse=True)
    elif sort_by == "year_asc":
        items.sort(key=lambda x: x["year"] or 9999)
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items


def run_search(q_text, mode="bert", top_k=12,
               genres=None, authors=None, sort_by="relevance",
               year_min=None, year_max=None):
    en_text, orig, lang = maybe_translate(q_text)
    q_tokens = preprocess(en_text)
    expansion = []
    t0 = time.time()

    pool_k = max(top_k * 5, 80)
    if mode == "tfidf":
        ranked = rank_tfidf(q_tokens, top_k=pool_k)
    elif mode == "rocchio":
        expanded = rocchio_expand(q_tokens)
        expansion = expanded[len(q_tokens):]
        ranked = rank_tfidf(expanded, top_k=pool_k)
    elif mode == "word2vec":
        expanded = w2v_expand(q_tokens)
        expansion = expanded[len(q_tokens):]
        ranked = rank_tfidf(expanded, top_k=pool_k)
    elif mode == "bert":
        if bert_model is None:
            return {"error": "BERT model not available"}
        cand = rank_tfidf(q_tokens, top_k=150)
        ranked = bert_rerank(en_text, cand, top_k=pool_k)
    else:
        return {"error": f"unknown mode: {mode}"}

    ranked = apply_filters(ranked, genres, authors, year_min, year_max)
    ranked = ranked[:top_k]
    items = make_items(ranked)
    items = sort_items(items, sort_by)

    elapsed = (time.time() - t0) * 1000.0
    return {
        "query": q_text,
        "translated_query": en_text if orig else None,
        "original_query": orig,
        "detected_lang": lang,
        "mode": mode,
        "tokens": q_tokens,
        "expansion": expansion,
        "elapsed_ms": round(elapsed, 2),
        "count": len(items),
        "results": items,
    }


# Enhanced suggestions generator
def generate_suggestions():
    """Generate diverse and interesting suggestions for users"""
    # Top genres in the corpus
    genre_counter = Counter()
    for gl in df["genre_list"]:
        for g in (gl or []):
            if g:
                genre_counter[g] += 1

    top_genres = [g for g, _ in genre_counter.most_common(10)]

    # Generate genre-specific suggestions
    genre_suggestions = [
        f"books in {genre}" for genre in top_genres[:5]
    ]

    # Author suggestions
    author_counter = Counter(df["author"].dropna().astype(str))
    top_authors = [a for a, _ in author_counter.most_common(8)]

    author_suggestions = [
        f"books by {author}" for author in top_authors[:5]
    ]

    # Theme-based suggestions
    theme_suggestions = [
        "science fiction space exploration",
        "mystery detective thriller",
        "romantic love story",
        "historical fiction",
        "fantasy magic adventure",
        "horror supernatural",
        "biography true story",
        "self help personal growth"
    ]

    # Combine all suggestions
    all_suggestions = genre_suggestions + author_suggestions + theme_suggestions

    return all_suggestions


# Generate suggestions
SUGGESTIONS_EN = generate_suggestions()[:15]  # Limit to 15 suggestions
SUGGESTIONS_AR = [
    "رواية خيال علمي",
    "قصة رعب",
    "مغامرات فانتازيا",
    "رحلة بحث عن الحقيقة",
    "كتاب تطوير ذات"
]

TOP_GENRES = [g for g, _ in Counter(genre for gl in df["genre_list"] for genre in (gl or [])).most_common(50)]
TOP_AUTHORS = [a for a, _ in Counter(df["author"].dropna().astype(str)).most_common(80)]

app = Flask(__name__, static_folder=HERE, static_url_path="")
CORS(app)


@app.route("/")
def home():
    return send_from_directory(HERE, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "ready": True})


@app.route("/api/stats")
def stats():
    return jsonify({
        "documents": int(len(df)),
        "unique_terms": int(len(inverted)),
        "avg_doc_len": float(df["tokens"].apply(len).mean()),
        "has_word2vec": w2v is not None,
        "has_bert": bert_model is not None,
        "modes": ["tfidf", "rocchio"] +
                 (["word2vec"] if w2v is not None else []) +
                 (["bert"] if bert_model is not None else []),
        "top_genres": TOP_GENRES,
        "top_authors": TOP_AUTHORS,
        "suggestions": SUGGESTIONS_EN
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    p = request.get_json(force=True, silent=True) or {}
    q = (p.get("query") or "").strip()
    mode = (p.get("mode") or ("bert" if bert_model is not None else "word2vec")).strip().lower()
    try:
        top_k = int(p.get("top_k", 12))
    except (TypeError, ValueError):
        top_k = 12
    top_k = max(1, min(top_k, 50))
    genres = p.get("genres") or []
    authors = p.get("authors") or []
    sort_by = (p.get("sort_by") or "relevance").lower()
    year_min = p.get("year_min")
    year_max = p.get("year_max")
    try:
        year_min = int(year_min) if year_min else None
        year_max = int(year_max) if year_max else None
    except (TypeError, ValueError):
        year_min = year_max = None

    if not q:
        return jsonify({"error": "empty query", "results": []}), 400

    return jsonify(run_search(q, mode, top_k, genres, authors, sort_by, year_min, year_max))


@app.route("/api/similar/<int:doc_id>")
def api_similar(doc_id):
    try:
        top_k = int(request.args.get("top_k", 10))
    except (TypeError, ValueError):
        top_k = 10
    top_k = max(1, min(top_k, 30))
    pairs = find_similar(doc_id, top_k=top_k)
    items = make_items(pairs)
    return jsonify({"doc_id": doc_id, "count": len(items), "results": items})


@app.route("/api/document/<int:doc_id>")
def api_document(doc_id):
    if doc_id not in DOC_INDEX:
        return jsonify({"error": "not found"}), 404
    row = DOC_INDEX[doc_id]
    title = str(row["title"])
    author = str(row["author"]) if pd.notna(row["author"]) else "Unknown"
    return jsonify({
        "doc_id": int(row["doc_id"]),
        "title": title,
        "author": author,
        "year": parse_year(str(row.get("pub_date", ""))),
        "genres": list(row["genre_list"]) if isinstance(row["genre_list"], list) else [],
        "summary": str(row["summary"]),
        "cover_url": fetch_cover(title, author),
    })


print(f"[ready] {len(df):,} documents, {len(inverted):,} unique terms, "
      f"w2v={'yes' if w2v else 'no'}, bert={'yes' if bert_model else 'no'}")
print("=" * 70)

if __name__ == "__main__":
    print("\nserving on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)