"""
embedding_features.py
=====================
Static embedding features for Arabic diacritization project.
Teammate B: Word & Character Embeddings
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. FASTTEXT EMBEDDINGS
# ============================================================================

def load_fasttext_embeddings(path: str, limit: int = 100000) -> Dict[str, np.ndarray]:
    """
    Load FastText Arabic embeddings from .vec file.
    
    Inputs:
        path (str): Path to FastText .vec file (e.g., "cc.ar.300.vec")
        limit (int): Maximum number of vectors to load (for memory)
    
    Outputs:
        Dict[str, np.ndarray]: Dictionary mapping words to 300D vectors
                               Returns empty dict if file not found
    
    Example:
        fasttext = load_fasttext_embeddings("cc.ar.300.vec", limit=50000)
        vector = fasttext["الكتاب"]  # shape: (300,)
    """
    print(f"Loading FastText from {path}...")
    
    if not os.path.exists(path):
        print(f"Error: FastText file not found at {path}")
        return {}
    
    word_vectors = {}
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read header: vocab_size dim
            header = f.readline().strip()
            if not header:
                print("Error: Empty FastText file")
                return {}
            
            parts = header.split()
            if len(parts) != 2:
                print("Error: Invalid FastText format")
                return {}
            
            vocab_size, dim = int(parts[0]), int(parts[1])
            print(f"FastText: {vocab_size:,} words, {dim} dimensions")
            
            # Read vectors
            for i, line in enumerate(f):
                if i >= limit:
                    break
                
                parts = line.rstrip().split(' ', 1)
                if len(parts) != 2:
                    continue
                
                word = parts[0]
                vector = np.fromstring(parts[1], sep=' ', dtype=np.float32)
                
                if len(vector) == dim:
                    word_vectors[word] = vector
        
        print(f"✓ Loaded {len(word_vectors):,} FastText vectors")
        return word_vectors
        
    except Exception as e:
        print(f"Error loading FastText: {e}")
        return {}


def get_fasttext_word_vectors(sentence: str, 
                             fasttext_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Get FastText vectors for all words in a sentence.
    
    Inputs:
        sentence (str): Arabic sentence (e.g., "ذهب علي إلى الشاطئ")
        fasttext_dict (Dict[str, np.ndarray]): FastText dictionary from load_fasttext_embeddings()
    
    Outputs:
        np.ndarray: Array of shape (num_words, 300) containing word vectors
                    Unknown words get zero vectors
    
    Example:
        fasttext = load_fasttext_embeddings("cc.ar.300.vec")
        vectors = get_fasttext_word_vectors("السلام عليكم", fasttext)
        # vectors.shape: (2, 300)
    """
    words = sentence.strip().split()
    vectors = []
    
    if not fasttext_dict:
        print("Warning: FastText dictionary is empty")
        return np.zeros((len(words), 300))
    
    for word in words:
        if word in fasttext_dict:
            vectors.append(fasttext_dict[word])
        else:
            # Get vector dimension from first word in dict
            dim = 300 if not fasttext_dict else len(next(iter(fasttext_dict.values())))
            vectors.append(np.zeros(dim))
    
    return np.array(vectors)


# ============================================================================
# 2. CHARACTER EMBEDDINGS
# ============================================================================

# def create_trainable_char_embeddings(vocab: List[str], 
#                                     embedding_dim: int = 64) -> Tuple[np.ndarray, Dict[str, int]]:
#     """
#     Create trainable character embeddings.
    
#     Inputs:
#         vocab (List[str]): List of unique Arabic characters
#                           (e.g., ['ا', 'ب', 'ت', 'ث', ...])
#         embedding_dim (int): Dimension of character embeddings (default: 64)
    
#     Outputs:
#         Tuple[np.ndarray, Dict[str, int]]:
#             - embedding_matrix: numpy array of shape (vocab_size, embedding_dim)
#             - char_to_idx: Dictionary mapping characters to indices
    
#     Example:
#         arabic_chars = ['ا', 'ب', 'ت', 'ث', 'ج']
#         emb_matrix, char_map = create_trainable_char_embeddings(arabic_chars, 50)
#         # emb_matrix.shape: (7, 50)  # +2 for special tokens
#         # char_map: {'ا': 0, 'ب': 1, 'ت': 2, 'ث': 3, 'ج': 4, '<PAD>': 5, '<UNK>': 6}
#     """
#     # Sort for consistency
#     sorted_chars = sorted(set(vocab))
    
#     # Create mappings
#     char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
    
#     # Add special tokens
#     special_tokens = ['<PAD>', '<UNK>']
#     for token in special_tokens:
#         char_to_idx[token] = len(char_to_idx)
    
#     # Create embedding matrix
#     vocab_size = len(char_to_idx)
#     np.random.seed(42)  # For reproducibility
#     embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1
    
#     # Zero vector for padding
#     embedding_matrix[char_to_idx['<PAD>']] = np.zeros(embedding_dim)
    
#     print(f"✓ Created character embeddings: {vocab_size} chars → {embedding_dim}D")
#     return embedding_matrix, char_to_idx


# def get_character_vectors(text: str, 
#                          char_to_idx: Dict[str, int],
#                          embedding_matrix: np.ndarray) -> np.ndarray:
#     """
#     Get character vectors for text.
    
#     Inputs:
#         text (str): Arabic text (e.g., "ذهب")
#         char_to_idx (Dict[str, int]): Character mapping from create_trainable_char_embeddings()
#         embedding_matrix (np.ndarray): Embedding matrix from create_trainable_char_embeddings()
    
#     Outputs:
#         np.ndarray: Array of shape (len(text), embedding_dim) containing character vectors
    
#     Example:
#         chars = ['ذ', 'ه', 'ب']
#         emb_matrix, char_map = create_trainable_char_embeddings(chars, 32)
#         vectors = get_character_vectors("ذهب", char_map, emb_matrix)
#         # vectors.shape: (3, 32)
#     """
#     indices = []
    
#     for char in text:
#         if char in char_to_idx:
#             indices.append(char_to_idx[char])
#         else:
#             indices.append(char_to_idx.get('<UNK>', 0))
    
#     return embedding_matrix[indices]


# # ============================================================================
# # 3. BAG OF WORDS & TF-IDF (Static Count-based Features)
# # ============================================================================

# def create_bow_features(sentences: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
#     """
#     Create Bag of Words features.
    
#     Inputs:
#         sentences (List[str]): List of Arabic sentences
#                               (e.g., ["ذهب علي", "الكتاب على الطاولة"])
    
#     Outputs:
#         Tuple[np.ndarray, Dict[str, int]]:
#             - bow_matrix: numpy array of shape (num_sentences, vocab_size)
#             - vocab_dict: Dictionary mapping words to indices
    
#     Example:
#         sentences = ["ذهب علي", "الكتاب على الطاولة"]
#         bow_matrix, vocab = create_bow_features(sentences)
#         # bow_matrix.shape: (2, 6)  # 2 sentences, 6 unique words
#         # bow_matrix[0]: [1, 1, 0, 0, 0, 0]  # "ذهب" and "علي" present
#     """
#     from collections import Counter
    
#     # Build vocabulary
#     all_words = []
#     for sentence in sentences:
#         words = sentence.strip().split()
#         all_words.extend(words)
    
#     # Count word frequencies
#     word_counts = Counter(all_words)
    
#     # Create vocabulary dictionary
#     vocab_dict = {word: idx for idx, word in enumerate(word_counts.keys())}
    
#     # Create BoW matrix
#     num_sentences = len(sentences)
#     vocab_size = len(vocab_dict)
#     bow_matrix = np.zeros((num_sentences, vocab_size), dtype=np.int32)
    
#     for i, sentence in enumerate(sentences):
#         words = sentence.strip().split()
#         for word in words:
#             if word in vocab_dict:
#                 bow_matrix[i, vocab_dict[word]] += 1
    
#     print(f"✓ Created BoW features: {num_sentences} sentences × {vocab_size} words")
#     return bow_matrix, vocab_dict


# def create_tfidf_features(sentences: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
#     """
#     Create TF-IDF features.
    
#     Inputs:
#         sentences (List[str]): List of Arabic sentences
    
#     Outputs:
#         Tuple[np.ndarray, Dict[str, int]]:
#             - tfidf_matrix: numpy array of shape (num_sentences, vocab_size)
#             - vocab_dict: Dictionary mapping words to indices
    
#     Example:
#         sentences = ["ذهب علي", "الكتاب على الطاولة", "علي يقرأ الكتاب"]
#         tfidf_matrix, vocab = create_tfidf_features(sentences)
#         # tfidf_matrix.shape: (3, 7)
#         # Values are TF-IDF scores (floats)
#     """
#     from collections import Counter
    
#     # Build vocabulary and document frequencies
#     vocab = set()
#     doc_freq = Counter()  # Number of documents containing each word
    
#     for sentence in sentences:
#         words = set(sentence.strip().split())
#         vocab.update(words)
#         doc_freq.update(words)
    
#     # Create vocabulary dictionary
#     vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
#     # Create TF-IDF matrix
#     num_sentences = len(sentences)
#     vocab_size = len(vocab)
#     tfidf_matrix = np.zeros((num_sentences, vocab_size))
    
#     # Total number of documents
#     N = num_sentences
    
#     for i, sentence in enumerate(sentences):
#         words = sentence.strip().split()
#         total_words = len(words)
        
#         if total_words == 0:
#             continue
        
#         # Count term frequency in this document
#         tf_counter = Counter(words)
        
#         for word, count in tf_counter.items():
#             if word in vocab_dict:
#                 idx = vocab_dict[word]
                
#                 # Term Frequency
#                 tf = count / total_words
                
#                 # Inverse Document Frequency
#                 idf = np.log(N / (1 + doc_freq[word]))
                
#                 # TF-IDF
#                 tfidf_matrix[i, idx] = tf * idf
    
#     print(f"✓ Created TF-IDF features: {num_sentences} sentences × {vocab_size} words")
#     return tfidf_matrix, vocab_dict


# # ============================================================================
# # 4. SIMPLE WORD EMBEDDINGS (Fallback)
# # ============================================================================

# def create_simple_word_embeddings(vocab: List[str], 
#                                  embedding_dim: int = 100) -> Dict[str, np.ndarray]:
#     """
#     Create simple deterministic word embeddings (fallback when FastText not available).
    
#     Inputs:
#         vocab (List[str]): List of Arabic words
#         embedding_dim (int): Dimension of embeddings (default: 100)
    
#     Outputs:
#         Dict[str, np.ndarray]: Dictionary mapping words to random but deterministic vectors
    
#     Example:
#         words = ["الكتاب", "القلم", "المدرسة"]
#         embeddings = create_simple_word_embeddings(words, 50)
#         # embeddings["الكتاب"].shape: (50,)
#     """
#     np.random.seed(42)
#     word_vectors = {}
    
#     for word in vocab:
#         # Create deterministic embedding from word hash
#         hash_val = abs(hash(word)) % (2**32)
#         np.random.seed(hash_val)
#         word_vectors[word] = np.random.randn(embedding_dim) * 0.1
    
#     print(f"✓ Created simple embeddings for {len(vocab)} words ({embedding_dim}D)")
#     return word_vectors


# # ============================================================================
# # 5. UTILITY FUNCTIONS
# # ============================================================================

# def calculate_oov_rate(sentences: List[str], 
#                       embeddings_dict: Dict[str, np.ndarray]) -> float:
#     """
#     Calculate Out-Of-Vocabulary (OOV) rate.
    
#     Inputs:
#         sentences (List[str]): List of Arabic sentences
#         embeddings_dict (Dict[str, np.ndarray]): Embeddings dictionary (FastText or simple)
    
#     Outputs:
#         float: OOV rate as percentage (0-100)
    
#     Example:
#         fasttext = load_fasttext_embeddings("cc.ar.300.vec", limit=10000)
#         oov_rate = calculate_oov_rate(["ذهب علي إلى الشاطئ"], fasttext)
#         # Returns: 0.0 (if all words found) or higher
#     """
#     total_words = 0
#     oov_words = 0
    
#     for sentence in sentences:
#         words = sentence.strip().split()
#         total_words += len(words)
#         for word in words:
#             if word not in embeddings_dict:
#                 oov_words += 1
    
#     if total_words == 0:
#         return 100.0
    
#     oov_rate = (oov_words / total_words) * 100
#     return oov_rate


# def save_embeddings(embeddings_dict: Dict[str, np.ndarray], 
#                    filename: str = "embeddings.pkl"):
#     """
#     Save embeddings to disk.
    
#     Inputs:
#         embeddings_dict (Dict[str, np.ndarray]): Embeddings dictionary
#         filename (str): Output filename
    
#     Outputs:
#         None (saves file to disk)
    
#     Example:
#         fasttext = load_fasttext_embeddings("cc.ar.300.vec", limit=50000)
#         save_embeddings(fasttext, "arabic_fasttext.pkl")
#     """
#     with open(filename, 'wb') as f:
#         pickle.dump(embeddings_dict, f)
    
#     print(f"✓ Saved embeddings to {filename}")


# def load_embeddings(filename: str = "embeddings.pkl") -> Dict[str, np.ndarray]:
#     """
#     Load embeddings from disk.
    
#     Inputs:
#         filename (str): Input filename
    
#     Outputs:
#         Dict[str, np.ndarray]: Embeddings dictionary
    
#     Example:
#         embeddings = load_embeddings("arabic_fasttext.pkl")
#     """
#     with open(filename, 'rb') as f:
#         embeddings_dict = pickle.load(f)
    
#     print(f"✓ Loaded embeddings from {filename} ({len(embeddings_dict)} words)")
#     return embeddings_dict


# # ============================================================================
# # 6. MAIN DEMO FUNCTION
# # ============================================================================

# def demo_all_features():
#     """
#     Demonstrate all embedding features.
    
#     Inputs: None
#     Outputs: Prints demonstration of all features
#     """
#     print("=" * 60)
#     print("ARABIC EMBEDDING FEATURES DEMO")
#     print("=" * 60)
    
#     # Sample data
#     sentences = [
#         "ذهب علي إلى الشاطئ",
#         "الكتاب على الطاولة",
#         "الطالب يقرأ الدرس"
#     ]
    
#     # 1. FastText embeddings
#     print("\n1. FASTTEXT EMBEDDINGS")
#     print("-" * 40)
    
#     # Try to load FastText (if file exists)
#     fasttext_path = "cc.ar.300.vec"
#     if os.path.exists(fasttext_path):
#         fasttext = load_fasttext_embeddings(fasttext_path, limit=50000)
        
#         if fasttext:
#             # Get vectors for a sentence
#             test_sentence = "ذهب علي"
#             vectors = get_fasttext_word_vectors(test_sentence, fasttext)
#             print(f"Test sentence: '{test_sentence}'")
#             print(f"FastText vectors shape: {vectors.shape}")
#             print(f"OOV rate: {calculate_oov_rate(sentences, fasttext):.2f}%")
#     else:
#         print(f"FastText file not found at: {fasttext_path}")
#         print("Skipping FastText demo...")
    
#     # 2. Character embeddings
#     print("\n2. CHARACTER EMBEDDINGS")
#     print("-" * 40)
    
#     # Extract characters from sentences
#     all_chars = set()
#     for sentence in sentences:
#         for char in sentence:
#             if char.strip():  # Skip spaces
#                 all_chars.add(char)
    
#     chars = list(all_chars)
#     emb_matrix, char_map = create_trainable_char_embeddings(chars, embedding_dim=32)
    
#     print(f"Character vocabulary size: {len(char_map)}")
#     print(f"Embedding matrix shape: {emb_matrix.shape}")
    
#     # Test character encoding
#     test_word = "ذهب"
#     char_vectors = get_character_vectors(test_word, char_map, emb_matrix)
#     print(f"Character vectors for '{test_word}': shape = {char_vectors.shape}")
    
#     # 3. Bag of Words
#     print("\n3. BAG OF WORDS FEATURES")
#     print("-" * 40)
    
#     bow_matrix, bow_vocab = create_bow_features(sentences)
#     print(f"BoW matrix shape: {bow_matrix.shape}")
#     print(f"Vocabulary size: {len(bow_vocab)}")
#     print(f"Sample BoW vector (first sentence): {bow_matrix[0, :5]}...")
    
#     # 4. TF-IDF
#     print("\n4. TF-IDF FEATURES")
#     print("-" * 40)
    
#     tfidf_matrix, tfidf_vocab = create_tfidf_features(sentences)
#     print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
#     print(f"Vocabulary size: {len(tfidf_vocab)}")
#     print(f"Sample TF-IDF vector (first sentence): {tfidf_matrix[0, :5]}...")
    
#     # 5. Simple embeddings (fallback)
#     print("\n5. SIMPLE WORD EMBEDDINGS")
#     print("-" * 40)
    
#     # Extract vocabulary
#     vocab = set()
#     for sentence in sentences:
#         words = sentence.strip().split()
#         vocab.update(words)
    
#     simple_embeddings = create_simple_word_embeddings(list(vocab), embedding_dim=50)
#     print(f"Created {len(simple_embeddings)} simple embeddings")
#     print(f"Vector for 'الكتاب': shape = {simple_embeddings['الكتاب'].shape}")
    
#     print("\n" + "=" * 60)
#     print("✓ DEMO COMPLETE - ALL FEATURES WORKING!")
#     print("=" * 60)
    
#     return {
#         'fasttext': fasttext if 'fasttext' in locals() else None,
#         'char_embeddings': (emb_matrix, char_map),
#         'bow_features': (bow_matrix, bow_vocab),
#         'tfidf_features': (tfidf_matrix, tfidf_vocab),
#         'simple_embeddings': simple_embeddings
#     }


# if __name__ == "__main__":
#     # Run the demo
#     results = demo_all_features()