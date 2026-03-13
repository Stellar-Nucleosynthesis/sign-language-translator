import os

import numpy as np
import spacy
from pathlib import Path

from dotenv import load_dotenv
from gensim import corpora, models

load_dotenv("train_lsa.env")
INPUT_FILE = os.getenv("INPUT_FILE")
SAVE_DIR = os.getenv("SAVE_DIR")
NUM_TOPICS = int(os.getenv("NUM_TOPICS", 100))

nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text: str):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens

def tokenize(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    articles = [a.strip() for a in content.split("\n\n") if a.strip()]
    return articles


def prepare_texts(articles):
    texts = []
    for article in articles:
        tokens = lemmatize_text(article)
        texts.append(tokens)
    return texts


def train_lsa(texts, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(
        corpus_tfidf,
        id2word=dictionary,
        num_topics=NUM_TOPICS
    )

    dictionary.save(str(save_dir / "dictionary.dict"))
    tfidf.save(str(save_dir / "tfidf.model"))
    lsi.save(str(save_dir / "lsa.model"))

    return lsi, dictionary, tfidf

def preprocess_query(query, dictionary, tfidf):
    tokens = lemmatize_text(query)
    bow = dictionary.doc2bow(tokens)
    tfidf_vec = tfidf[bow]
    return tfidf_vec

def lsa_to_dense(vec, num_topics):
    dense = np.zeros(num_topics)
    for topic_id, value in vec:
        dense[topic_id] = value
    return dense

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    articles = load_articles(INPUT_FILE)
    texts = prepare_texts(articles)

    lsi, dictionary, tfidf = train_lsa(texts, SAVE_DIR)

    queries = [
        "Cats eating fruit",
        "Kittens ingesting apples",
        "Mark studying computers"
    ]
    input_vecs = [preprocess_query(q, dictionary, tfidf) for q in queries]
    output_vecs = [lsa_to_dense(lsi[v], lsi.num_topics) for v in input_vecs]

    for i in range(len(output_vecs)):
        for j in range(i + 1, len(output_vecs)):
            sim = cosine_similarity(output_vecs[i], output_vecs[j])
            print(f"Similarity({queries[i]} , {queries[j]}) = {sim:.4f}")


if __name__ == "__main__":
    main()