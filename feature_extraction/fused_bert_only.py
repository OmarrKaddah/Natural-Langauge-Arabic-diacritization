import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz

from feature_extraction.bert_embedding_extraction.build_features import build_word_features


PREPROCESSED_PATH = "train_processed.json"
OUT_DIR = "artifacts_bert_only/"


def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d["undiac"] for d in data]


def build_bert_only_features():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading sentences...")
    sentences = load_sentences(PREPROCESSED_PATH)
    print("Total:", len(sentences))

    # Storage
    all_rows = []       # character-level metadata
    all_labels = []     # labels
    word_embeddings = []  # BERT word-level

    print("\nEncoding BERT for each sentence...")

    for s_idx, sent in enumerate(tqdm(sentences)):
        words, bert_embs = build_word_features(sent)   # bert_embs = list of 768-d arrays

        word_embeddings.append(bert_embs)

        # Build rows (fake structure for compatibility)
        for w_idx, word in enumerate(words):
            for c_idx, ch in enumerate(word):
                all_rows.append({
                    "sentence_index": s_idx,
                    "word_index": w_idx,
                    "char": ch,
                    "char_index": c_idx
                })
                all_labels.append(0)  # placeholder (no labels here)

    # Broadcast 768-d word embeddings to characters
    print("\nBroadcasting BERT to characters...")

    n_chars = len(all_rows)
    X = np.zeros((n_chars, 768), dtype=np.float32)

    for i, r in enumerate(all_rows):
        s = r["sentence_index"]
        w = r["word_index"]
        X[i] = word_embeddings[s][w]

    print("Final matrix:", X.shape)

    # Convert to CSR sparse
    X_sparse = csr_matrix(X)

    save_npz(os.path.join(OUT_DIR, "X_bert_only.npz"), X_sparse)
    np.save(os.path.join(OUT_DIR, "y_bert_only.npy"), np.array(all_labels))

    print("\nSaved:")
    print(" → artifacts_bert_only/X_bert_only.npz")
    print(" → artifacts_bert_only/y_bert_only.npy")


if __name__ == "__main__":
    build_bert_only_features()
