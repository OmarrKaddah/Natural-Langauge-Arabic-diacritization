import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, save_npz

from feature_extraction.classical.build_classical_features import build_all_features
from feature_extraction.bert_embedding_extraction.build_features import build_word_features


PREPROCESSED_PATH = "train_processed.json"
OUT_DIR = "artifacts_bert_classical/"


def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d["undiac"] for d in data]


def build_bert_classical():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nSTEP 1: Loading classical features...")
    X_classical, y, rows = build_all_features(PREPROCESSED_PATH, out_dir=OUT_DIR)
    print("Classical shape:", X_classical.shape)

    print("\nSTEP 2: Loading sentences...")
    sentences = load_sentences(PREPROCESSED_PATH)

    print("\nSTEP 3: BERT encoding...")
    word_embeds = []

    for s_idx, sent in enumerate(tqdm(sentences)):
        words, bert_embs = build_word_features(sent)
        word_embeds.append(bert_embs)

    # Broadcast BERT to characters
    print("\nBroadcasting BERT to chars...")
    fused_dim = 768
    dense_word_feats = np.zeros((len(rows), fused_dim), dtype=np.float32)

    for i, r in enumerate(rows):
        s = r["sentence_index"]
        w = r["word_index"]
        dense_word_feats[i] = word_embeds[s][w]

    # Sparse conversion
    X_word = csr_matrix(dense_word_feats)

    # Final fusion
    X_fused = hstack([X_classical, X_word], format="csr")
    print("\nFinal fused shape:", X_fused.shape)

    save_npz(os.path.join(OUT_DIR, "X_train_bert_classical.npz"), X_fused)
    np.save(os.path.join(OUT_DIR, "y_train_bert_classical.npy"), y)

    print("\nSaved to artifacts_bert_classical/")


if __name__ == "__main__":
    build_bert_classical()
