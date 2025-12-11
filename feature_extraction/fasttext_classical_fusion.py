import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, save_npz
from gensim.models import KeyedVectors

# ===========================
# 1) Classical (Reduced) Features
# ===========================
# You MUST provide this file:
# feature_extraction/classical/build_less_features.py
from feature_extraction.classical.build_less_classical_features import build_less_features


# ===========================
# CONFIG
# ===========================
PREPROCESSED_PATH = "train_processed.json"
FASTTEXT_PATH = "cc.ar.300.vec"   # 🔥 CHANGE this to full path
OUT_DIR = "artifacts_fasttext_classical/"
FASTTEXT_DIM = 300
REDUCED_K = 700   # classical dimensionality (reasonable)


# ===========================
# HELPERS
# ===========================
def load_fasttext_model(path):
    """
    Loads FastText .vec file using gensim.
    """
    print(f"\nLoading FastText model from: {path}")
    model = KeyedVectors.load_word2vec_format(path, binary=False)
    print("✔ FastText loaded.")
    return model


def fasttext_vector(word, ft_model, dim=300):
    """
    Returns FastText embedding for a word, or zeros if OOV.
    """
    if word in ft_model.key_to_index:
        return ft_model[word]
    return np.zeros(dim, dtype=np.float32)


def load_undiac_sentences(preprocessed_path):
    """
    Returns list of undiacritized sentences in same order as classical rows.
    """
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["undiac"].strip() for item in data if "undiac" in item]


# ===========================
# MAIN PIPELINE
# ===========================
def build_fasttext_classical_fused(
        preprocessed_path=PREPROCESSED_PATH,
        fasttext_path=FASTTEXT_PATH,
        out_dir=OUT_DIR,
        reduced_k=REDUCED_K):

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*70)
    print(" STEP 1 — Classical (Reduced) Features")
    print("="*70)

    X_classical, y, rows = build_less_features(
        preprocessed_path=preprocessed_path,
        out_dir=out_dir,
        k_best=reduced_k
    )

    n_chars = X_classical.shape[0]
    print(f"\nClassical feature shape: {X_classical.shape}")
    print(f"Total character rows:   {n_chars}")

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 2 — Load Sentences")
    print("="*70)

    sentences = load_undiac_sentences(preprocessed_path)
    print(f"Loaded {len(sentences)} sentences")

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 3 — Load FastText Model")
    print("="*70)

    ft = load_fasttext_model(fasttext_path)

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 4 — Build FastText per Sentence Word")
    print("="*70)

    sentence_word_fasttext = []

    for sent in tqdm(sentences, desc="FastText encoding"):
        words = sent.split()  # simple whitespace tokenizer
        fast_embs = [fasttext_vector(w, ft, FASTTEXT_DIM) for w in words]

        sentence_word_fasttext.append({
            "words": words,
            "embeddings": fast_embs
        })

    print("\n✔ Finished FastText embedding extraction.")

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 5 — Broadcast FastText to Characters")
    print("="*70)

    dense_fasttext = np.zeros((n_chars, FASTTEXT_DIM), dtype=np.float32)

    for i, row in enumerate(rows):
        s = row["sentence_index"]
        w = row["word_index"]

        fast_vecs = sentence_word_fasttext[s]["embeddings"]

        if w >= len(fast_vecs):
            raise IndexError(f"Word index {w} invalid for sentence {s}")

        dense_fasttext[i, :] = fast_vecs[w]

    print("✔ FastText broadcast complete.")
    print("FastText matrix shape:", dense_fasttext.shape)

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 6 — Concatenate Classical + FastText")
    print("="*70)

    X_fast = csr_matrix(dense_fasttext)
    X_fused = hstack([X_classical, X_fast], format="csr")

    print("\nFinal fused shape:", X_fused.shape)
    print("Labels shape:", y.shape)

    # ----------------------------------------
    print("\n" + "="*70)
    print(" STEP 7 — Save Outputs")
    print("="*70)

    save_npz(f"{out_dir}/X_train_fasttext_classical.npz", X_fused)
    np.save(f"{out_dir}/y_train_fasttext_classical.npy", y)

    # Save metadata
    with open(f"{out_dir}/rows_meta.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n✔ Saved all outputs to:", out_dir)


# Run if executed directly
if __name__ == "__main__":
    build_fasttext_classical_fused()
