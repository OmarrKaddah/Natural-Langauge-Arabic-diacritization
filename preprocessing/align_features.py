# preprocessing/align_features.py

def align_word_features_to_chars(chars, words, word_features):
    """
    Expands word-level features to character-level features.
    """
    pos_seq = []
    gender_seq = []
    number_seq = []
    aspect_seq = []

    char_idx = 0

    for word, feats in zip(words, word_features):
        for _ in word:
            pos_seq.append(feats["pos"])
            gender_seq.append(feats["gender"])
            number_seq.append(feats["number"])
            aspect_seq.append(feats["aspect"])
            char_idx += 1

        # handle space between words (if present)
        if char_idx < len(chars) and chars[char_idx] == " ":
            pos_seq.append("SPACE")
            gender_seq.append("SPACE")
            number_seq.append("SPACE")
            aspect_seq.append("SPACE")
            char_idx += 1

    assert len(pos_seq) == len(chars)

    return pos_seq, gender_seq, number_seq, aspect_seq
