import json
import pickle
from tqdm import tqdm
import numpy as np

# ==========================================
# CUSTOM FASTTEXT LOADER
# ==========================================
from feature_extraction.fasttext.fasttext import (
    load_fasttext_embeddings,
    get_fasttext_word_vectors
)

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
# BERT WORD EMBEDDING EXTRACTION
# ==========================================
from feature_extraction.bert_embedding_extraction.build_features import build_word_features


# ==========================================
# MAIN PIPELINE
# ==========================================
def combine_bert_fasttext():

    print("Loading dataset...")
    sentences = load_train_sentences("train_processed.json")
    print("Total sentences in dataset:", len(sentences))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 🔥 PROCESS ONLY FIRST 10 SENTENCES
    sentences = sentences[:10]
    print("Processing only first 10 sentences")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ----------------------------------
    # Load FastText
    # ----------------------------------
    print("Loading FastText embeddings...")
    ft = load_fasttext_embeddings(
        "C:/Users/ziada/Downloads/cc.ar.300.vec/cc.ar.300.vec", 
        limit=50000
    )
    print("FastText loaded successfully.")

    json_output = []
    pkl_output = []

    for sent in tqdm(sentences, desc="Processing sentences"):

        words, bert_embs = build_word_features(sent)

        fast_embs = get_fasttext_word_vectors(sent, ft)

        if len(fast_embs) != len(words):
            fast_embs = [ft.get(w, np.zeros(300)) for w in words]

        combined = [
            np.concatenate([bert_embs[i], fast_embs[i]])
            for i in range(len(words))
        ]

        print("\n=== SAMPLE CHECK ===")
        print("Sentence:", sent)
        print("Words:", words[:10])
        print("BERT shape:", bert_embs[0].shape)
        print("FastText shape:", fast_embs[0].shape)
        print("Combined shape:", combined[0].shape)
        print("====================\n")

        json_output.append({
            "sentence": sent,
            "words": words,
            "embeddings": [vec.tolist() for vec in combined]
        })

        pkl_output.append({
            "sentence": sent,
            "words": words,
            "embeddings": combined
        })

    with open("combined_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    with open("combined_embeddings.pkl", "wb") as f:
        pickle.dump(pkl_output, f)

    print("\n✔ Saved combined embeddings to:")
    print("→ combined_embeddings.json")
    print("→ combined_embeddings.pkl")
    print("Done!")


if __name__ == "__main__":
    combine_bert_fasttext()
