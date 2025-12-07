import numpy as np
from scipy.sparse import csr_matrix

def build_scalar_features(rows):
    """Output example: each row corresponds to a character and contains:
    [is_first_in_word, is_last_in_word, word_length]
    ك  ت  ا  ب
     1  0  4
     0  0  4
     0  0  4
     0  1  4
"""
    data = []

    for r in rows:
        data.append([
            1 if r["pos_in_word"] == 0 else 0,                    # is_first
            1 if r["pos_in_word"] == r["word_length"] - 1 else 0, # is_last
            r["word_length"]                                      # word length
        ])

    return csr_matrix(np.array(data, dtype=float))
