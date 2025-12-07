# quick_start.py
from fasttext import *

# 1. Load FastText (if you have the file)
fasttext = load_fasttext_embeddings("C:/Users/ziada/Downloads/cc.ar.300.vec/cc.ar.300.vec", limit=50000)

# 2. Get word vectors for a sentence
sentence = "ذهب علي إلى الشاطئ"
if fasttext:
    vectors = get_fasttext_word_vectors(sentence, fasttext)
    for i in range(len(vectors)):
        print(f"Word: {sentence.split()[i]}, Vector: {vectors[i][:5]}...")  # Print first 5 dimensions
    print(f"Word vectors shape: {vectors.shape}")  # (5, 300)

# 3. Create character embeddings
# chars = ['ذ', 'ه', 'ب', 'ع', 'ل', 'ي', 'إ', 'ل', 'ى']
# char_matrix, char_map = create_trainable_char_embeddings(chars, 64)
# char_vectors = get_character_vectors("ذهب", char_map, char_matrix)
# print(f"Character vectors shape: {char_vectors.shape}")  # (3, 64)

# # 4. Create BoW features
# sentences = ["ذهب علي", "الكتاب على الطاولة"]
# bow_matrix, bow_vocab = create_bow_features(sentences)
# print(f"BoW matrix shape: {bow_matrix.shape}")  # (2, vocab_size)