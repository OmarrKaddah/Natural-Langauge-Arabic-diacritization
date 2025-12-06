from feature_extraction.bert_embedding_extraction.load_dataset import load_train_sentences
from feature_extraction.bert_embedding_extraction.build_features import build_word_features

print("===== TESTING WORD-LEVEL BERT FROM train_processed.json =====")

# Load JSON dataset
sentences = load_train_sentences("train_processed.json")

print("Loaded", len(sentences), "sentences.")

# Use the first sentence
sentence = sentences[0]
print("\nSentence:")
print(sentence)

words, word_embs = build_word_features(sentence)

print("\nWords:", words)
print("Number of words:", len(words))
print("Number of BERT word embeddings:", len(word_embs))

# Validate match
if len(words) != len(word_embs):
    print("\n❌ ERROR: Mismatch between words and embeddings!")
else:
    print("\n✔ Correct: one embedding per word.")

# Show first few
for w, e in zip(words[:10], word_embs[:10]):
    print(f"\nWord: {w}")
    print("Embedding (first 10 dims):", e[:10])
