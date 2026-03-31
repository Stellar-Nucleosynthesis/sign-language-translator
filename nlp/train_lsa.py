import logging
import os
import re
import time
import random
import multiprocessing
from pathlib import Path

import numpy as np
import spacy
from dotenv import load_dotenv
from gensim import corpora, models

load_dotenv("train_lsa.env")

INPUT_DIR    = os.getenv("INPUT_DIR")
SAVE_DIR     = os.getenv("SAVE_DIR")
NUM_TOPICS   = int(os.getenv("NUM_TOPICS",   100))
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES",   100_000))

DICT_PATH = Path(SAVE_DIR) / "dictionary.dict"
TFIDF_PATH = Path(SAVE_DIR) / "tfidf.model"
LSI_PATH = Path(SAVE_DIR) / "lsa.model"

NLP_BATCH_SIZE  = 1000
NLP_N_PROCESS   = max(1, multiprocessing.cpu_count() - 1)
DICT_CHUNK_SIZE = 50_000
LSI_CHUNK_SIZE   = 20_000
PRINT_EVERY_PCT = 5

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

doc_start_re = re.compile(r"<doc.*?>")
doc_end_re   = re.compile(r"</doc>")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("gensim").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:   return f"{h}h {m:02d}m {s:02d}s"
    if m:   return f"{m}m {s:02d}s"
    return f"{s}s"


def _progress(label: str, n: int, total: int | None, start: float, extra: str = ""):
    elapsed = time.time() - start
    rate    = n / elapsed if elapsed else 0
    if total:
        pct = n / total * 100
        eta = (total - n) / rate if rate else 0
        print(
            f"  {label}  {pct:5.1f}%  ({n:,} / {total:,})"
            f"  |  {rate:,.0f} doc/s  |  ETA {_fmt_time(eta)}"
            + (f"  |  {extra}" if extra else ""),
            flush=True,
        )
    else:
        print(
            f"  {label}  {n:,} docs"
            f"  |  {rate:,.0f} doc/s  |  elapsed {_fmt_time(elapsed)}"
            + (f"  |  {extra}" if extra else ""),
            flush=True,
        )


def _raw_article_stream(root_dir: Path):
    for path in sorted(root_dir.rglob("wiki_*")):
        with open(path, "r", encoding="utf-8") as f:
            lines: list[str] = []
            inside = False
            for line in f:
                if doc_start_re.search(line):
                    inside = True
                    lines  = []
                elif doc_end_re.search(line):
                    inside = False
                    yield " ".join(lines)
                    lines  = []
                elif inside:
                    lines.append(line.strip())


def reservoir_sample(root_dir: Path, k: int) -> list[str]:
    print(f"\n  Sampling {k:,} random articles from dump …", flush=True)
    reservoir: list[str] = []
    start = time.time()
    n = 0

    for n, article in enumerate(_raw_article_stream(root_dir), start=1):
        if n <= k:
            reservoir.append(article)
        else:
            j = random.randint(0, n - 1)
            if j < k:
                reservoir[j] = article

        if n % 1_000_000 == 0:
            _progress("Scanning", n, None, start)

    random.shuffle(reservoir)
    elapsed = time.time() - start
    print(
        f"  Sampling done — {k:,} selected from {n:,} total"
        f"  ({k/n:.1%})  |  {_fmt_time(elapsed)}",
        flush=True,
    )
    return reservoir


def _doc_to_tokens(doc) -> list[str]:
    return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]


def token_stream(source):
    articles = _raw_article_stream(Path(source)) if isinstance(source, (str, Path)) else iter(source)
    for spacy_doc in nlp.pipe(articles, batch_size=NLP_BATCH_SIZE, n_process=NLP_N_PROCESS):
        yield _doc_to_tokens(spacy_doc)


def build_dictionary(source, total: int | None):
    print(f"\n[1/4] Building dictionary …", flush=True)
    start      = time.time()
    dictionary = corpora.Dictionary()
    chunk: list[list[str]] = []
    n          = 0
    next_pct   = PRINT_EVERY_PCT

    for tokens in token_stream(source):
        chunk.append(tokens)
        n += 1

        if len(chunk) >= DICT_CHUNK_SIZE:
            dictionary.add_documents(chunk)
            chunk = []

        if total:
            pct = n / total * 100
            if pct >= next_pct:
                _progress("Dictionary", n, total, start, f"vocab {len(dictionary):,}")
                next_pct += PRINT_EVERY_PCT

    if chunk:
        dictionary.add_documents(chunk)

    elapsed = time.time() - start
    print(f"  Done — {n:,} docs | vocab {len(dictionary):,} | {_fmt_time(elapsed)}", flush=True)
    return dictionary, n


def serialize_bow_corpus(source, save_dir: Path, dictionary, total: int | None):
    corpus_path = str(save_dir / "corpus.mm")
    print(f"\n[2/4] Serialising BoW corpus …", flush=True)
    start    = time.time()
    next_pct = PRINT_EVERY_PCT

    def _bow_gen():
        nonlocal next_pct
        n = 0
        for tokens in token_stream(source):
            n += 1
            if total:
                pct = n / total * 100
                if pct >= next_pct:
                    _progress("BoW", n, total, start)
                    next_pct += PRINT_EVERY_PCT
            yield dictionary.doc2bow(tokens)

    corpora.MmCorpus.serialize(corpus_path, _bow_gen())
    elapsed = time.time() - start
    print(f"  Done — {_fmt_time(elapsed)} — saved to {corpus_path}", flush=True)
    return corpus_path


def train_lsa(root_dir, save_dir):
    wall_start = time.time()
    save_dir   = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    limit_note = f"{MAX_ARTICLES:,} (random sample)" if MAX_ARTICLES else "all"
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          LSA Training  —  configuration              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Corpus dir  : {str(root_dir):<38}║")
    print(f"║  Output dir  : {str(save_dir):<38}║")
    print(f"║  Topics      : {str(NUM_TOPICS):<38}║")
    print(f"║  Articles    : {limit_note:<38}║")
    print(f"║  spaCy CPUs  : {str(NLP_N_PROCESS):<38}║")
    print("╚══════════════════════════════════════════════════════╝")

    if MAX_ARTICLES:
        source = reservoir_sample(Path(root_dir), MAX_ARTICLES)
        total  = len(source)
    else:
        source = root_dir
        total  = None

    dictionary, doc_count = build_dictionary(source, total)

    print(f"\n  Vocab before filtering : {len(dictionary):,}")
    dictionary.filter_extremes(no_below=10, no_above=0.4)
    print(f"  Vocab after  filtering : {len(dictionary):,}")

    dict_path = save_dir / "dictionary.dict"
    dictionary.save(str(dict_path))
    print(f"  Dictionary saved → {dict_path}")

    corpus_path = serialize_bow_corpus(source, save_dir, dictionary, total)
    bow_corpus  = corpora.MmCorpus(corpus_path)

    print("\n[3/4] Training TF-IDF …", flush=True)
    t0    = time.time()
    tfidf = models.TfidfModel(bow_corpus, dictionary=dictionary)
    print(f"  Done — {_fmt_time(time.time() - t0)}", flush=True)
    tfidf_path = save_dir / "tfidf.model"
    tfidf.save(str(tfidf_path))
    print(f"  TF-IDF saved → {tfidf_path}")

    print("\n[4/4] Training LSA …", flush=True)
    t0           = time.time()
    corpus_tfidf = tfidf[bow_corpus]
    lsi = models.LsiModel(
        corpus_tfidf,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        chunksize=LSI_CHUNK_SIZE,
    )
    print(f"  Done — {_fmt_time(time.time() - t0)}", flush=True)
    lsi_path = save_dir / "lsa.model"
    lsi.save(str(lsi_path))
    print(f"  LSA saved → {lsi_path}")

    total_elapsed = time.time() - wall_start
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Training complete                                   ║")
    print(f"║  Total time  : {_fmt_time(total_elapsed):<38}║")
    print(f"║  Documents   : {doc_count:<38,}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    return lsi, dictionary, tfidf


def preprocess_query(query, dictionary, tfidf):
    doc    = nlp(query)
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    return tfidf[dictionary.doc2bow(tokens)]


def lsa_to_dense(vec, num_topics):
    dense = np.zeros(num_topics)
    for topic_id, value in vec:
        dense[topic_id] = value
    return dense


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def main():
    # lsi, dictionary, tfidf = train_lsa(INPUT_DIR, SAVE_DIR)

    print("Loading dictionary...")
    dictionary = corpora.Dictionary.load(str(DICT_PATH))

    print("Loading TF-IDF model...")
    tfidf = models.TfidfModel.load(str(TFIDF_PATH))

    print("Loading LSA model...")
    lsi = models.LsiModel.load(str(LSI_PATH))

    print("Models loaded successfully")

    queries = [
        "Cats eating fruit",
        "Kittens ingesting apples",
        "Femboys studying transistors",
    ]
    input_vecs  = [preprocess_query(q, dictionary, tfidf) for q in queries]
    output_vecs = [lsa_to_dense(lsi[v], lsi.num_topics) for v in input_vecs]

    print("\n── Similarity results ──────────────────────────────────")
    for i in range(len(output_vecs)):
        for j in range(i + 1, len(output_vecs)):
            sim = cosine_similarity(output_vecs[i], output_vecs[j])
            print(f"  {queries[i]!r:35s}  ↔  {queries[j]!r:35s}  =  {sim:.4f}")
    print()


if __name__ == "__main__":
    main()