import json
import pickle
from tqdm import tqdm
import numpy as np
import fasttext

# ==========================================
# 1. LOAD DATASET
# ==========================================
def load_train_sentences(path="train_processed.json"):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item, dict) and "undiac" in item:
            sent = item["undiac"].strip()
            if sent:
                sentences.append(sent)
    return sentences


# ==========================================
# 2. FASTTEXT LOADING + VECTOR EXTRACTION
# ==========================================
FASTTEXT_PATH = "cc.ar.300.vec"   # CHANGE TO YOUR PATH

def load_fasttext_model(path=FASTTEXT_PATH):
    print("Loading FastText model...")
    model = fasttext.load_model(path)
    print("FastText loaded.")
    return model

def get_fasttext_word_vector(word, ft_model):
    """
    Returns a 300-dim FastText vector.
    """
    return ft_model.get_word_vector(word)  # numpy array


# ==========================================
# 3. BERT WORD EMBEDDING EXTRACTION
# ==========================================
from feature_extraction.bert_embedding_extraction.build_features import build_word_features


# ==========================================
# 4. MAIN COMBINED PIPELINE
# ==========================================
def combine_bert_fasttext():
    # ----------------------------------
    # Load dataset
    # ----------------------------------
    print("Loading dataset...")
    sentences = load_train_sentences("train_processed.json")
    print("Total sentences:", len(sentences))

    # ----------------------------------
    # Load FastText
    # ----------------------------------
    ft = load_fasttext_model(FASTTEXT_PATH)

    # ----------------------------------
    # Output containers
    # ----------------------------------
    json_output = []
    pkl_output = []

    # ----------------------------------
    # Process all sentences
    # ----------------------------------
    for sent in tqdm(sentences, desc="Processing sentences"):

        # Step A: BERT word-level embeddings
        words, bert_embs = build_word_features(sent)

        # Step B: FastText word embeddings
        fast_embs = [get_fasttext_word_vector(w, ft) for w in words]

        # Step C: Concatenate → 1068 dims
        combined = [
            np.concatenate([bert_embs[i], fast_embs[i]])   # 768 + 300
            for i in range(len(words))
        ]

        # ------------------------
        # VALIDATION CHECKS
        # ------------------------
        assert len(words) == len(bert_embs), "Mismatch: BERT"
        assert len(words) == len(fast_embs), "Mismatch: FastText"
        assert combined[0].shape[0] == 1068, "Concatenation incorrect"

        # Print first few only
        print("\n=== SAMPLE CHECK ===")
        print("Sentence:", sent)
        print("Words:", words[:10])
        print("BERT shape:", bert_embs[0].shape)
        print("FastText shape:", fast_embs[0].shape)
        print("Combined shape:", combined[0].shape)
        print("Total word vectors:", len(combined))
        print("====================\n")

        # ------------------------
        # Save for JSON
        # ------------------------
        json_output.append({
            "sentence": sent,
            "words": words,
            "embeddings": [vec.tolist() for vec in combined]
        })

        # ------------------------
        # Save for PKL
        # ------------------------
        pkl_output.append({
            "sentence": sent,
            "words": words,
            "embeddings": combined  # numpy arrays
        })

    # ==========================================
    # SAVE OUTPUT
    # ==========================================

    with open("combined_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    with open("combined_embeddings.pkl", "wb") as f:
        pickle.dump(pkl_output, f)

    print("\n✔ Saved combined embeddings to:")
    print("→ combined_embeddings.json")
    print("→ combined_embeddings.pkl")
    print("Done!")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    combine_bert_fasttext()
