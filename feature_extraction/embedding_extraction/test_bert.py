# test_bert.py

from preprocessing.preprocess_sentence import preprocess_sentence
from feature_extraction.embedding_extraction.build_features import build_features


sentence = "بَتَحَ"
undiac, chars, labels = preprocess_sentence(sentence)

embs = build_features(undiac, chars)

print("Characters:", chars)
print("Labels:", labels)
print("Number of embeddings:", len(embs[0]))

print("Embedding for ب:", embs[0][:10])   # print first 10 numbers
