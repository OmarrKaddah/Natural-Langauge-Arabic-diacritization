import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, save_npz
from gensim.models import KeyedVectors

from feature_extraction.classical.build_less_classical_features import build_all_features
from feature_extraction.bert_embedding_extraction.build_features import build_word_features

PREPROCESSED_PATH = "train_processed.json"
FASTTEXT_PATH = "cc.ar.300.vec"
OUT_DIR = "artifacts_full_fusion/"


def load_fasttext(path):
    print("Loading FastText:", path)
    return KeyedVectors.load_word2vec_format(path, binary=False)


def get_vec(word, model):
    return model[word] if word in model.key_to_index else np.zeros(300)


def load_sentences(p):
    return [d["undiac"].strip() for d in json.load(open(p, "r", encoding="utf8"))]


def build_full_fusion():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nSTEP1: Classical features...")
    X_classical, y, rows = build_all_features(PREPROCESSED_PATH, out_dir=OUT_DIR)

    print("\nSTEP2: Load sentences...")
    sentences = load_sentences(PREPROCESSED_PATH)

    print("\nSTEP3: Load FastText...")
    ft = load_fasttext(FASTTEXT_PATH)

    print("\nSTEP4: Compute BERT+FT word embeddings...")
    word_embs_all = []
    for s in tqdm(sentences):
        words, bert = build_word_features(s)
        ft_vecs = [get_vec(w, ft) for w in words]
        merged = [np.concatenate([bert[i], ft_vecs[i]]) for i in range(len(words))]
        word_embs_all.append(merged)

    print("\nSTEP5: Broadcast to characters...")
    fused_dim = 768 + 300
    dense = np.zeros((len(rows), fused_dim))

    for i, r in enumerate(rows):
        dense[i] = word_embs_all[r["sentence_index"]][r["word_index"]]

    X_word = csr_matrix(dense)

    print("\nSTEP6: Concatenate...")
    X_fused = hstack([X_classical, X_word], format="csr")
    print("Final shape:", X_fused.shape)

    save_npz(os.path.join(OUT_DIR, "X_train_full.npz"), X_fused)
    np.save(os.path.join(OUT_DIR, "y_train_full.npy"), y)

    print("\nSAVED → artifacts_full_fusion/")


if __name__ == "__main__":
    build_full_fusion()
