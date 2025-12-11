import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix, save_npz

# 1) Classical features
from feature_extraction.classical_features import build_all_features

# 2) BERT word-level embeddings
from feature_extraction.bert_embedding_extraction.build_features import build_word_features

# 3) FastText (.vec) word embeddings via gensim
from gensim.models import KeyedVectors


# ==========================================
# CONFIG
# ==========================================

PREPROCESSED_PATH = "train_processed.json"
FASTTEXT_PATH = "cc.ar.300.vec"   # 👈 change this to the *full path* of your cc.ar.300.vec
ARTIFACTS_DIR = "artifacts/"      # where to save final features


# ==========================================
# HELPERS
# ==========================================

def load_fasttext_model(path: str) -> KeyedVectors:
    """
    Load FastText word vectors from a .vec file using gensim.
    """
    print(f"Loading FastText (.vec) from: {path}")
    model = KeyedVectors.load_word2vec_format(path, binary=False)
    print("✅ FastText loaded.")
    return model


def get_fasttext_word_vector(word: str, ft_model: KeyedVectors, dim: int = 300) -> np.ndarray:
    """
    Returns a 300-dim FastText vector for a given word.
    If the word is OOV, returns a zero vector.
    """
    if word in ft_model.key_to_index:
        return ft_model[word]
    else:
        return np.zeros(dim, dtype=np.float32)


def load_undiac_sentences(preprocessed_path=PREPROCESSED_PATH):
    """
    Loads undiacritized sentences from train_processed.json
    in the same order as classical pipeline.
    """
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = [item["undiac"].strip() for item in data if isinstance(item, dict) and "undiac" in item]
    return sentences


# ==========================================
# MAIN FUSION PIPELINE
# ==========================================

def build_fused_features(preprocessed_path=PREPROCESSED_PATH,
                         fasttext_path=FASTTEXT_PATH,
                         out_dir=ARTIFACTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------
    # 1) Classical features (character-level)
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 1: Building classical character-level features")
    print("="*70)

    X_classical, y, rows = build_all_features(
        preprocessed_path=preprocessed_path,
        out_dir=out_dir
    )

    n_chars = X_classical.shape[0]
    print(f"\nClassical feature matrix shape: {X_classical.shape}")
    print(f"Number of character rows:       {len(rows)}")

    assert n_chars == len(rows), "Mismatch: rows vs X_classical rows"

    # --------------------------------------
    # 2) Load undiac sentences
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 2: Loading undiacritized sentences")
    print("="*70)

    sentences = load_undiac_sentences(preprocessed_path)
    print(f"Loaded {len(sentences)} sentences from {preprocessed_path}")

    # --------------------------------------
    # 3) Load FastText model
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 3: Loading FastText model")
    print("="*70)

    ft_model = load_fasttext_model(fasttext_path)

    # --------------------------------------
    # 4) Build BERT+FastText word embeddings per sentence
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 4: Building BERT+FastText word embeddings per sentence")
    print("="*70)

    # For each sentence index: store its words and [768+300] embeddings
    sentence_word_embeddings = []   # list of dicts: { "words": [...], "embeddings": [np.array(1068), ...] }

    for sent in tqdm(sentences, desc="Encoding sentences with BERT+FastText"):
        # BERT word-level
        words, bert_embs = build_word_features(sent)   # words: list[str], bert_embs: list/array of 768-d

        # FastText for each word
        fast_embs = [get_fasttext_word_vector(w, ft_model, dim=300) for w in words]

        # Concatenate
        combined = [
            np.concatenate([np.asarray(bert_embs[i]), fast_embs[i]], axis=0)
            for i in range(len(words))
        ]

        # Basic sanity check
        if len(words) > 0:
            assert combined[0].shape[0] == 768 + 300, "Concatenation dim mismatch (should be 1068)"

        sentence_word_embeddings.append({
            "words": words,
            "embeddings": combined
        })

    print("\n✅ Finished computing word-level BERT+FastText embeddings.")

    # --------------------------------------
    # 5) Broadcast word embeddings to each character row
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 5: Broadcasting BERT+FastText embeddings to characters")
    print("="*70)

    fused_dim = 768 + 300  # BERT (768) + FastText (300)
    dense_word_feats = np.zeros((n_chars, fused_dim), dtype=np.float32)

    for i, row in enumerate(rows):
        # Try multiple possible key names to be robust
        s_idx = row.get("sentence_index", row.get("sent_idx"))
        w_idx = row.get("word_index", row.get("w_idx"))

        if s_idx is None or w_idx is None:
            raise ValueError(
                f"Row {i} is missing 'sentence_index'/'word_index' "
                f"(got keys: {list(row.keys())})"
            )

        sent_info = sentence_word_embeddings[s_idx]

        # Safety: if something went wrong in tokenization alignment
        if w_idx >= len(sent_info["embeddings"]):
            raise IndexError(
                f"word_index {w_idx} out of range for sentence {s_idx} "
                f"(len(words)={len(sent_info['embeddings'])})"
            )

        dense_word_feats[i, :] = sent_info["embeddings"][w_idx]

    print(f"\nConstructed dense word-level feature matrix per character: {dense_word_feats.shape}")

    # --------------------------------------
    # 6) Concatenate classical + BERT+FastText
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 6: Concatenating classical + BERT+FastText features")
    print("="*70)

    X_word = csr_matrix(dense_word_feats)   # convert to sparse
    X_fused = hstack([X_classical, X_word], format="csr")

    print(f"\nFinal fused feature matrix shape: {X_fused.shape}")
    print(f"Labels shape:                     {y.shape}")

    # --------------------------------------
    # 7) Save fused features
    # --------------------------------------
    print("\n" + "="*70)
    print("STEP 7: Saving fused features")
    print("="*70)

    # Save as sparse matrix + labels
    save_npz(os.path.join(out_dir, "X_train_fused.npz"), X_fused)
    np.save(os.path.join(out_dir, "y_train.npy"), y)

    # Also save mapping rows → (sentence_index, word_index, char, word)
    meta_rows = [
        {
            "sentence_index": r.get("sentence_index", r.get("sent_idx")),
            "word_index": r.get("word_index", r.get("w_idx")),
            "char_index": r.get("char_index", r.get("c_idx")),
            "char": r.get("char"),
            "word": r.get("word"),
            "label": r.get("label"),
        }
        for r in rows
    ]
    with open(os.path.join(out_dir, "rows_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)

    # Optionally save a small sample of embeddings for debugging
    with open(os.path.join(out_dir, "sample_word_embeddings.pkl"), "wb") as f:
        # store only first 200 rows to keep file small
        sample = dense_word_feats[:200]
        pickle.dump(sample, f)

    print("\n✅ Saved:")
    print(f"  - {out_dir}X_train_fused.npz  (sparse fused features)")
    print(f"  - {out_dir}y_train.npy        (labels)")
    print(f"  - {out_dir}rows_meta.json     (character → sentence/word mapping)")
    print(f"  - {out_dir}sample_word_embeddings.pkl (debug sample)")
    print("\nAll done! 🚀")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    build_fused_features()
